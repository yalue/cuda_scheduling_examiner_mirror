/*
# Copyright (c) 2014-2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <memory>

#include <NVX/nvx.h>
#include <NVX/nvx_timer.hpp>

#include <NVX/Application.hpp>
#include <NVX/ConfigParser.hpp>
#include <OVX/FrameSourceOVX.hpp>
#include <OVX/RenderOVX.hpp>
#include <NVX/SyncTimer.hpp>
#include <OVX/UtilityOVX.hpp>

// for cuda_scheduling_examiner
#include <library_interface.h>
#include "third_party/cJSON.h"

namespace
{

// Utility
struct HoughTransformDemoParams
{
    vx_uint32   switchPeriod;
    vx_float32  scaleFactor;
    vx_enum     scaleType;
    vx_int32    CannyLowerThresh;
    vx_int32    CannyUpperThresh;
    vx_float32  dp;
    vx_float32  minDist;
    vx_uint32   minRadius;
    vx_uint32   maxRadius;
    vx_uint32   accThreshold;
    vx_uint32   circlesCapacity;
    vx_float32  rho;
    vx_float32  theta;
    vx_uint32   votesThreshold;
    vx_uint32   minLineLength;
    vx_uint32   maxLineGap;
    vx_uint32   linesCapacity;

    HoughTransformDemoParams()
        : switchPeriod(400),
        scaleFactor(.5f),
        scaleType(VX_INTERPOLATION_TYPE_BILINEAR),
        CannyLowerThresh(230),
        CannyUpperThresh(250),
        dp(2.f),
        minDist(10.f),
        minRadius(1),
        maxRadius(25),
        accThreshold(110),
        circlesCapacity(300),
        rho(1.f),
        theta(1.f),
        votesThreshold(100),
        minLineLength(25),
        maxLineGap(2),
        linesCapacity(300) {}
};

// Process events
struct EventData
{
    EventData(): showSource(true), stop(false), pause(false) {}

    bool showSource;
    bool stop;
    bool pause;
};

// Holds the local state for one instance of this benchmark.
typedef struct
{
    bool shouldRender;
    ovxio::ContextGuard *context;
    std::unique_ptr<ovxio::FrameSource> frameSource;
    std::unique_ptr<ovxio::Render> renderer;
    EventData eventData;
    vx_image frame;
    vx_image edges;
    vx_array circles;
    vx_array lines;
    vx_int32 numFrames;
    vx_node cvtNode;
    vx_node scaleDownNode;
    vx_node median3x3Node;
    vx_node equalizeHistNode;
    vx_node CannyNode;
    vx_node scaleUpNode;
    vx_node Sobel3x3Node;
    vx_node HoughCirclesNode;
    vx_node HoughSegmentsNode;
    vx_graph graph;
    HoughTransformDemoParams ht_params;
} BenchmarkState;

bool checkParams(vx_int32& CannyLowerThresh, vx_int32& CannyUpperThresh,
        vx_uint32& minRadius, vx_uint32& maxRadius, std::string & error)
{
    if (CannyLowerThresh > CannyUpperThresh)
    {
        error = "Inconsistent values of lower and upper Canny thresholds";
    }

    if (minRadius > maxRadius)
    {
        error = "Inconsistent minimum and maximum circle radius values";
    }

    return error.empty();
}


bool read(const std::string &configFile, HoughTransformDemoParams &config, std::string &error)
{
    const std::unique_ptr<nvxio::ConfigParser> parser(nvxio::createConfigParser());

    parser->addParameter("switchPeriod", nvxio::OptionHandler::unsignedInteger(
                &config.switchPeriod));
    parser->addParameter("scaleFactor", nvxio::OptionHandler::real(
                &config.scaleFactor,
                nvxio::ranges::moreThan(0.f)
                &
                nvxio::ranges::atMost(1.f)));
    parser->addParameter("scaleType", nvxio::OptionHandler::oneOf(
                &config.scaleType,
                { {"nearest",
                VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR},
                {"bilinear",
                VX_INTERPOLATION_TYPE_BILINEAR},
                {"area",
                VX_INTERPOLATION_TYPE_AREA},
                }));
    parser->addParameter("CannyLowerThresh", nvxio::OptionHandler::integer(
                &config.CannyLowerThresh,
                nvxio::ranges::moreThan(0)));
    parser->addParameter("CannyUpperThresh", nvxio::OptionHandler::integer(
                &config.CannyUpperThresh,
                nvxio::ranges::moreThan(0)));
    parser->addParameter("dp", nvxio::OptionHandler::real( &config.dp,
                nvxio::ranges::atLeast(1.f)));
    parser->addParameter("minDist", nvxio::OptionHandler::real( &config.minDist,
                nvxio::ranges::moreThan(0.f)));
    parser->addParameter("minRadius", nvxio::OptionHandler::unsignedInteger(
                &config.minRadius));
    parser->addParameter("maxRadius", nvxio::OptionHandler::unsignedInteger(
                &config.maxRadius,
                nvxio::ranges::moreThan(0u)));
    parser->addParameter("accThreshold", nvxio::OptionHandler::unsignedInteger(
                &config.accThreshold,
                nvxio::ranges::moreThan(0u)));
    parser->addParameter("circlesCapacity",
            nvxio::OptionHandler::unsignedInteger(
                &config.circlesCapacity,
                nvxio::ranges::moreThan(0u)
                &
                nvxio::ranges::atMost(1000u)));
    parser->addParameter("rho", nvxio::OptionHandler::real( &config.rho,
                nvxio::ranges::moreThan(0.f)));
    parser->addParameter("theta", nvxio::OptionHandler::real( &config.theta,
                nvxio::ranges::moreThan(0.f)
                &
                nvxio::ranges::atMost(180.f)));
    parser->addParameter("votesThreshold",
            nvxio::OptionHandler::unsignedInteger(
                &config.votesThreshold,
                nvxio::ranges::moreThan(0u)));
    parser->addParameter("minLineLength", nvxio::OptionHandler::unsignedInteger(
                &config.minLineLength,
                nvxio::ranges::moreThan(0u)));
    parser->addParameter("maxLineGap", nvxio::OptionHandler::unsignedInteger(
                &config.maxLineGap));
    parser->addParameter("linesCapacity", nvxio::OptionHandler::unsignedInteger(
                &config.linesCapacity,
                nvxio::ranges::moreThan(0u)
                &
                nvxio::ranges::atMost(1000u)));

    error = parser->parse(configFile);

    if (!error.empty())
    {
        return false;
    }

    return checkParams(config.CannyLowerThresh, config.CannyUpperThresh,
            config.minRadius, config.maxRadius, error);
}


void keyboardEventCallback(void* eventData, vx_char key, vx_uint32 /*x*/, vx_uint32 /*y*/)
{
    EventData* data = static_cast<EventData*>(eventData);
    if (key == 27) // escape
    {
        data->stop = true;
    }
    else if (key == 'm')
    {
        data->showSource = !data->showSource;
    }
    else if (key == 32) // space
    {
        data->pause = !data->pause;
    }
}

}

static void Cleanup(void *data)
{
    BenchmarkState *state = (BenchmarkState *)data;
    if (state->cvtNode) vxReleaseNode(&state->cvtNode);
    if (state->scaleDownNode) vxReleaseNode(&state->scaleDownNode);
    if (state->median3x3Node) vxReleaseNode(&state->median3x3Node);
    if (state->equalizeHistNode) vxReleaseNode(&state->equalizeHistNode);
    if (state->CannyNode) vxReleaseNode(&state->CannyNode);
    if (state->scaleUpNode) vxReleaseNode(&state->scaleUpNode);
    if (state->Sobel3x3Node) vxReleaseNode(&state->Sobel3x3Node);
    if (state->HoughCirclesNode) vxReleaseNode(&state->HoughCirclesNode);
    if (state->HoughSegmentsNode) vxReleaseNode(&state->HoughSegmentsNode);
    if (state->graph) vxReleaseGraph(&state->graph);
    if (state->frame) vxReleaseImage(&state->frame);
    if (state->edges) vxReleaseImage(&state->edges);
    if (state->circles) vxReleaseArray(&state->circles);
    if (state->lines) vxReleaseArray(&state->lines);
    if (state->frameSource) delete state->frameSource.get();
    if (state->renderer) delete state->renderer.get();
    if (state->context) delete state->context;
    memset(state, 0, sizeof(*state));
    free(state);
}

static int initFrameSource(BenchmarkState *state, std::string sourceUri)
{
    // Create a NVXIO-based frame source
    state->frameSource = ovxio::createDefaultFrameSource(*(state->context), sourceUri);

    if (!state->frameSource || !state->frameSource->open())
    {
        std::cerr << "Error: Can't open source URI " << sourceUri << std::endl;
        return 0;
    }

    ovxio::FrameSource::Parameters frameConfig = state->frameSource->getConfiguration();

    if ((frameConfig.frameWidth * state->ht_params.scaleFactor < 16) ||
            (frameConfig.frameHeight * state->ht_params.scaleFactor < 16))
    {
        std::cerr << "Error: Scale factor is too small" << std::endl;
        return 0;
    }
    return 1;
}

static int initRender(BenchmarkState *state)
{
    if (!state->shouldRender)
    {
        return 1;
    }
    ovxio::FrameSource::Parameters frameConfig = state->frameSource->getConfiguration();
    // Create a NVXIO-based render
    state->renderer = ovxio::createDefaultRender(*(state->context),
            "Hough Transform Demo", frameConfig.frameWidth, frameConfig.frameHeight);

    if (!state->renderer)
    {
        std::cerr << "Error: Cannot create render!" << std::endl;
        return 0;
    }

    // The application recieves the keyboard events via the
    // keyboardEventCallback() function registered via the renderer object

    state->eventData = EventData();
    state->renderer->setOnKeyboardEventCallback(keyboardEventCallback, &state->eventData);
    return 1;
}

static int initGraph(BenchmarkState *state)
{
    ovxio::FrameSource::Parameters frameConfig = state->frameSource->getConfiguration();
    // Create OpenVX objects
    // frame and edges vx_image objects are created to hold the frames from
    // the video source and the output of the Canny edge detector respectively

    state->frame = vxCreateImage(*(state->context), frameConfig.frameWidth,
            frameConfig.frameHeight, VX_DF_IMAGE_RGBX);
    NVXIO_CHECK_REFERENCE(state->frame);

    state->edges = vxCreateImage(*(state->context), frameConfig.frameWidth,
            frameConfig.frameHeight, VX_DF_IMAGE_U8);
    NVXIO_CHECK_REFERENCE(state->edges);

    // Similary the lines and circles vx_array objects are created to hold
    // the output from Hough line-segment detector and Hough circle detector

    state->circles = vxCreateArray(*(state->context), NVX_TYPE_POINT3F, state->ht_params.circlesCapacity);
    NVXIO_CHECK_REFERENCE(state->circles);

    state->lines = vxCreateArray(*(state->context), NVX_TYPE_POINT4F, state->ht_params.linesCapacity);
    NVXIO_CHECK_REFERENCE(state->lines);

    // Create OpenVX Threshold to hold Canny thresholds

    vx_threshold CannyThreshold = vxCreateThreshold(*(state->context), VX_THRESHOLD_TYPE_RANGE, VX_TYPE_INT32);
    NVXIO_CHECK_REFERENCE(CannyThreshold);
    NVXIO_SAFE_CALL( vxSetThresholdAttribute(CannyThreshold,
                VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER,
                &state->ht_params.CannyLowerThresh,
                sizeof(state->ht_params.CannyLowerThresh)) );
    NVXIO_SAFE_CALL( vxSetThresholdAttribute(CannyThreshold,
                VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER,
                &state->ht_params.CannyUpperThresh,
                sizeof(state->ht_params.CannyUpperThresh)) );

    // vxCreateGraph() instantiates the pipeline

    state->graph = vxCreateGraph(*(state->context));
    NVXIO_CHECK_REFERENCE(state->graph);

    // Virtual images for internal processing

    vx_image virt_U8 = vxCreateVirtualImage(state->graph, 0, 0, VX_DF_IMAGE_U8);
    NVXIO_CHECK_REFERENCE(virt_U8);

    vx_image virt_scaled = vxCreateVirtualImage(state->graph,
            static_cast<vx_uint32>(frameConfig.frameWidth *
                state->ht_params.scaleFactor),
            static_cast<vx_uint32>(frameConfig.frameHeight *
                state->ht_params.scaleFactor), VX_DF_IMAGE_U8);
    NVXIO_CHECK_REFERENCE(virt_scaled);

    vx_image virt_blurred = vxCreateVirtualImage(state->graph, 0, 0, VX_DF_IMAGE_U8);
    NVXIO_CHECK_REFERENCE(virt_blurred);

    vx_image virt_equalized = vxCreateVirtualImage(state->graph, 0, 0, VX_DF_IMAGE_U8);
    NVXIO_CHECK_REFERENCE(virt_equalized);

    vx_image virt_edges = vxCreateVirtualImage(state->graph, 0, 0, VX_DF_IMAGE_U8);
    NVXIO_CHECK_REFERENCE(virt_edges);

    vx_image virt_dx = vxCreateVirtualImage(state->graph, 0, 0, VX_DF_IMAGE_S16);
    NVXIO_CHECK_REFERENCE(virt_dx);

    vx_image virt_dy = vxCreateVirtualImage(state->graph, 0, 0, VX_DF_IMAGE_S16);
    NVXIO_CHECK_REFERENCE(virt_dy);

    // Node creation
    // The frame is converted to grayscale by converting from RGB to YUV and
    // extracting the Y channel and scaled down (scale factor is defined in
    // ht_params.scaleType and defaults to 0.5). Frame is then blurred with
    // a median3x3 filter and has histogram equalized. Output is passed to
    // the Canny edge detector and Sobel 3x3 filter. The output is then passed
    // through nvxHoughSegmentsNode() and nvxHoughCirclesNode() to detect line
    // segments and circles. The detected line segments and circles are scaled
    // up back by ht_params.scaleType

    state->cvtNode = vxColorConvertNode(state->graph, state->frame, virt_U8);
    NVXIO_CHECK_REFERENCE(state->cvtNode);

    state->scaleDownNode = vxScaleImageNode(state->graph, virt_U8, virt_scaled,
            state->ht_params.scaleType);
    NVXIO_CHECK_REFERENCE(state->scaleDownNode);

    state->median3x3Node = vxMedian3x3Node(state->graph, virt_scaled, virt_blurred);
    NVXIO_CHECK_REFERENCE(state->median3x3Node);

    state->equalizeHistNode = vxEqualizeHistNode(state->graph, virt_blurred, virt_equalized);
    NVXIO_CHECK_REFERENCE(state->equalizeHistNode);

    state->CannyNode = vxCannyEdgeDetectorNode(state->graph, virt_equalized,
            CannyThreshold, 3, VX_NORM_L1, virt_edges);
    NVXIO_CHECK_REFERENCE(state->CannyNode);

    state->scaleUpNode = vxScaleImageNode(state->graph, virt_edges,
            state->edges, state->ht_params.scaleType);
    NVXIO_CHECK_REFERENCE(state->scaleUpNode);

    state->Sobel3x3Node = vxSobel3x3Node(state->graph, virt_equalized, virt_dx, virt_dy);
    NVXIO_CHECK_REFERENCE(state->Sobel3x3Node);

    state->HoughCirclesNode = nvxHoughCirclesNode(state->graph, virt_edges, virt_dx, virt_dy,
            state->circles, nullptr,  state->ht_params.dp, state->ht_params.minDist,
            state->ht_params.minRadius, state->ht_params.maxRadius, state->ht_params.accThreshold);
    NVXIO_CHECK_REFERENCE(state->HoughCirclesNode);

    state->HoughSegmentsNode = nvxHoughSegmentsNode(state->graph, virt_edges,
            state->lines, state->ht_params.rho, state->ht_params.theta,
            state->ht_params.votesThreshold, state->ht_params.minLineLength,
            state->ht_params.maxLineGap, nullptr);
    NVXIO_CHECK_REFERENCE(state->HoughSegmentsNode);

    // Release virtual images (the graph will hold references internally)

    vxReleaseImage(&virt_U8);
    vxReleaseImage(&virt_scaled);
    vxReleaseImage(&virt_blurred);
    vxReleaseImage(&virt_equalized);
    vxReleaseImage(&virt_edges);
    vxReleaseImage(&virt_dx);
    vxReleaseImage(&virt_dy);

    // Release Threshold object (the graph will hold references internally)

    vxReleaseThreshold(&CannyThreshold);

    // Ensure highest graph optimization level

    const char* option = "-O3";
    NVXIO_SAFE_CALL( vxSetGraphAttribute(state->graph, NVX_GRAPH_VERIFY_OPTIONS, option, strlen(option)) );

    // Verify the graph

    vx_status verify_status = vxVerifyGraph(state->graph);
    if (verify_status != VX_SUCCESS)
    {
        std::cerr << "Error: Graph verification failed. See the NVX LOG for explanation." << std::endl;
        return 0;
    }
    return 1;
}

static int processConfig(BenchmarkState *state, char *info)
{
    if (!info) // take the default setting
    {
        state->shouldRender = false;
        return 1;
    }
    cJSON *parsed = cJSON_Parse(info);
    cJSON *entry = NULL;
    if (!parsed || (parsed->type != cJSON_Object))
    {
        std::cerr << "Error: Wrong format of additional_info in the configuration file" << std::endl;
        goto ErrorCleanup;
    }
    entry = cJSON_GetObjectItem(parsed, "shouldRender");
    if (!entry || (entry->type != cJSON_True && entry->type != cJSON_False))
    {
        state->shouldRender = false;
    }
    else
    {
        state->shouldRender = entry->type == cJSON_True;
    }
    return 1;
ErrorCleanup:
    if (parsed) cJSON_Delete(parsed);
    return 0;
}

static void* Initialize(InitializationParameters *params)
{
    BenchmarkState *state = NULL;
    state = (BenchmarkState *) malloc(sizeof(*state));
    if (!state) return NULL;
    memset(state, 0, sizeof(*state));

    if (!processConfig(state, params->additional_info))
    {
        Cleanup(state);
        return NULL;
    }

    nvxio::Application &app = nvxio::Application::get();

    // Parse command line arguments
    std::string sourceUri = app.findSampleFilePath("signs.avi");
    std::string configFile = app.findSampleFilePath("hough_transform_demo_config.ini");

    app.init(1, NULL);

    state->ht_params = HoughTransformDemoParams();
    // Read and check input parameters
    std::string error;
    if (!read(configFile, state->ht_params, error))
    {
        std::cerr << error << std::endl;
        Cleanup(state);
        return NULL;
    }

    state->ht_params.theta *= ovxio::PI_F / 180.0f; // convert to radians

    // NVXIO-based renderer object and frame source are instantiated
    // and attached to the OpenVX context object. NVXIO ContextGuard
    // object is used to automatically manage the OpenVX context
    // creation and destruction.
    state->context = new ovxio::ContextGuard;
    vxDirective(*(state->context), VX_DIRECTIVE_ENABLE_PERFORMANCE);

    // Messages generated by the OpenVX framework will be given
    // to ovxio::stdoutLogCallback
    vxRegisterLogCallback(*(state->context), &ovxio::stdoutLogCallback, vx_false_e);

    if (!initFrameSource(state, sourceUri))
    {
        Cleanup(state);
        return NULL;
    }

    if (!initRender(state))
    {
        Cleanup(state);
        return NULL;
    }

    if (!initGraph(state))
    {
        Cleanup(state);
        return NULL;
    }

    return state;
}

static int CopyIn(void *data)
{
    return 1;
}

// The main function call of hough_transform demo creates the object of type
// Application (defined in NVXIO library). Command line arguments are parsed
// and the input video filename is read into sourceURI and the configuration
// parameters are read into the configFile
static int Execute(void *data)
{
    try
    {
        BenchmarkState *state = (BenchmarkState *)data;

loop:
        // Main loop
        // When it's not rendered, pause isn't an option. The frame is processed
        // as it goes.
        // When it's rendered, need to check whether it's paused.
        if (!state->shouldRender || (state->shouldRender && !state->eventData.pause))
        {
            // The main processing loop simply reads the input frames using
            // the source->fetch(). The rendering code in the main loop decides
            // whether to display orignal (unprocessed) source image or edges
            // overlayed with detected lines and circles based on user input

            ovxio::FrameSource::FrameStatus frameStatus = state->frameSource->fetch(state->frame);

            if (frameStatus == ovxio::FrameSource::TIMEOUT)
            {
                std::cerr << "Error: frame source featch TIMEOUT" << std::endl;
                goto loop;
            }

            if (frameStatus == ovxio::FrameSource::CLOSED)
            {
                if (!state->frameSource->open())
                {
                    std::cerr << "Error: Failed to reopen the source" << std::endl;
                    return 0;
                }
                goto loop;
            }

            // Process
            NVXIO_SAFE_CALL( vxProcessGraph(state->graph) );

            // Scale detected circles
            vx_size num_circles = 0;
            NVXIO_SAFE_CALL( vxQueryArray(state->circles, VX_ARRAY_ATTRIBUTE_NUMITEMS, &num_circles, sizeof(num_circles)) );

            if (num_circles > 0)
            {
                vx_map_id map_id;
                void *ptr;
                vx_size stride;
                NVXIO_SAFE_CALL( vxMapArrayRange(state->circles, 0, num_circles, &map_id, &stride, &ptr, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0) );

                for (vx_size i = 0; i < num_circles; ++i)
                {
                    nvx_point3f_t *c = (nvx_point3f_t *)vxFormatArrayPointer(ptr, i, stride);
                    c->x /= state->ht_params.scaleFactor;
                    c->y /= state->ht_params.scaleFactor;
                    c->z /= state->ht_params.scaleFactor;
                }

                NVXIO_SAFE_CALL( vxUnmapArrayRange(state->circles, map_id) );
            }

            // Scale detected lines (convert it to array with start and end coordinates)
            vx_size lines_count = 0;
            NVXIO_SAFE_CALL( vxQueryArray(state->lines, VX_ARRAY_ATTRIBUTE_NUMITEMS, &lines_count, sizeof(lines_count)) );

            if (lines_count > 0)
            {
                vx_map_id map_id;
                vx_size stride;
                void *ptr;
                NVXIO_SAFE_CALL( vxMapArrayRange(state->lines, 0, lines_count, &map_id, &stride, &ptr, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0) );

                for (vx_size i = 0; i < lines_count; ++i)
                {
                    nvx_point4f_t *coord = (nvx_point4f_t *)vxFormatArrayPointer(ptr, i, stride);

                    coord->x /= state->ht_params.scaleFactor;
                    coord->y /= state->ht_params.scaleFactor;
                    coord->z /= state->ht_params.scaleFactor;
                    coord->w /= state->ht_params.scaleFactor;
                }

                NVXIO_SAFE_CALL( vxUnmapArrayRange(state->lines, map_id) );
            }

            // switch image/edges view every switchPeriod-th frame
            if (state->ht_params.switchPeriod > 0)
            {
                state->numFrames++;
                if (state->numFrames % state->ht_params.switchPeriod == 0)
                {
                    state->eventData.showSource = !state->eventData.showSource;
                }
            }

        }

        // Show original image or detected edges
        if (state->shouldRender && state->renderer)
        {
            if (state->eventData.showSource)
            {
                state->renderer->putImage(state->frame);
            }
            else
            {
                state->renderer->putImage(state->edges);
            }

            // Draw detected circles
            ovxio::Render::CircleStyle circleStyle = { { 255u, 0u, 255u, 255u}, 2 };
            state->renderer->putCircles(state->circles, circleStyle);

            // Draw detected lines
            ovxio::Render::LineStyle lineStyle = { { 0u, 255u, 255u, 255u}, 2 };
            state->renderer->putLines(state->lines, lineStyle);

            // Flush all rendering commands
            if (!state->renderer->flush())
            {
                state->eventData.stop = true;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 0;
    }

    return 1;
}

static int CopyOut(void *data, TimingInformation *times)
{
    times->kernel_count = 0;
    return 1;
}

static const char* GetName(void)
{
    return "VisionWorks demo: hough transform";
}

// This should be the only function we export from the library, to provide
// pointers to all of the other functions.
int RegisterFunctions(BenchmarkLibraryFunctions *functions)
{
    functions->initialize = Initialize;
    functions->copy_in = CopyIn;
    functions->execute = Execute;
    functions->copy_out = CopyOut;
    functions->cleanup = Cleanup;
    functions->get_name = GetName;
    return 1;
}
