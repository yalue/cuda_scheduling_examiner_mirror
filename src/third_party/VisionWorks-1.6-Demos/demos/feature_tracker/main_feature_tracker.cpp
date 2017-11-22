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

#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <memory>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <NVX/nvx.h>
#include <NVX/nvx_timer.hpp>

#include "feature_tracker.hpp"
#include <NVX/Application.hpp>
#include <NVX/ConfigParser.hpp>
#include <OVX/FrameSourceOVX.hpp>
#include <OVX/RenderOVX.hpp>
#include <NVX/SyncTimer.hpp>
#include <OVX/UtilityOVX.hpp>

// for cuda_scheduling_examiner
#include <library_interface.h>
#include "third_party/cJSON.h"

// Process events
struct EventData
{
    EventData(): shouldStop(false), pause(false) {}

    bool shouldStop;
    bool pause;
};

// Holds the local state for one instance of this benchmark.
typedef struct
{
    EventData eventData;
    std::unique_ptr<ovxio::Render> renderer;
    std::unique_ptr<ovxio::FrameSource> source;
    vx_delay frame_delay;
    vx_image frame;
    vx_image prevFrame;
    vx_image mask;
    nvx::FeatureTracker *tracker;
    ovxio::ContextGuard *context;
    bool shouldRender;
} BenchmarkState;

static void eventCallback(void* eventData, vx_char key, vx_uint32, vx_uint32)
{
    EventData* data = static_cast<EventData*>(eventData);

    if (key == 27)
    {
        data->shouldStop = true;
    }
    else if (key == 32)
    {
        data->pause = !data->pause;
    }
}

static void displayState(ovxio::Render *renderer,
        const ovxio::FrameSource::Parameters &sourceParams,
        nvx::FeatureTracker::Params &config,
        double proc_ms, double total_ms)
{
    std::ostringstream txt;

    txt << std::fixed << std::setprecision(1);

    ovxio::Render::TextBoxStyle style = {{255, 255, 255, 255}, {0, 0, 0, 127}, {10, 10}};

    txt << "Source size: " << sourceParams.frameWidth << 'x' << sourceParams.frameHeight << std::endl;
    txt << "Detector: " << (config.use_harris_detector ? "Harris" : "FAST") << std::endl;
    txt << "Algorithm: " << proc_ms << " ms / " << 1000.0 / proc_ms << " FPS" << std::endl;
    txt << "Display: " << total_ms  << " ms / " << 1000.0 / total_ms << " FPS" << std::endl;

    txt << std::setprecision(6);
    txt.unsetf(std::ios_base::floatfield);
    txt << "LIMITED TO " << nvxio::Application::get().getFPSLimit() << " FPS FOR DISPLAY" << std::endl;

    txt << "Space - pause/resume" << std::endl;
    txt << "Esc - close the demo";
    renderer->putTextViewport(txt.str(), style);
}

static bool read(const std::string & nf, nvx::FeatureTracker::Params &config, std::string &message)
{
    std::unique_ptr<nvxio::ConfigParser> ftparser(nvxio::createConfigParser());

    ftparser->addParameter("pyr_levels", nvxio::OptionHandler::unsignedInteger(&config.pyr_levels,
                nvxio::ranges::atLeast(1u)
                &
                nvxio::ranges::atMost(8u)));
    ftparser->addParameter("lk_win_size",
            nvxio::OptionHandler::unsignedInteger(&config.lk_win_size,
                nvxio::ranges::atLeast(3u)
                &
                nvxio::ranges::atMost(32u)));
    ftparser->addParameter("lk_num_iters",
            nvxio::OptionHandler::unsignedInteger(&config.lk_num_iters,
                nvxio::ranges::atLeast(1u)
                &
                nvxio::ranges::atMost(100u)));
    ftparser->addParameter("array_capacity",
            nvxio::OptionHandler::unsignedInteger(&config.array_capacity,
                nvxio::ranges::atLeast(1u)));
    ftparser->addParameter("detector_cell_size",
            nvxio::OptionHandler::unsignedInteger(&config.detector_cell_size,
                nvxio::ranges::atLeast(1u)));
    ftparser->addParameter("detector",
            nvxio::OptionHandler::oneOf(&config.use_harris_detector,
                { {"harris", true},
                {"fast", false} }));
    ftparser->addParameter("harris_k",
            nvxio::OptionHandler::real(&config.harris_k,
                nvxio::ranges::moreThan(0.0f)));
    ftparser->addParameter("harris_thresh",
            nvxio::OptionHandler::real(&config.harris_thresh,
                nvxio::ranges::moreThan(0.0f)));
    ftparser->addParameter("fast_type",
            nvxio::OptionHandler::unsignedInteger(&config.fast_type,
                nvxio::ranges::atLeast(9u)
                &
                nvxio::ranges::atMost(12u)));
    ftparser->addParameter("fast_thresh",
            nvxio::OptionHandler::unsignedInteger(&config.fast_thresh,
                nvxio::ranges::lessThan(255u)));

    message = ftparser->parse(nf);

    return message.empty();
}

static void Cleanup(void *data)
{
    BenchmarkState *state = (BenchmarkState *)data;
    if (state->mask) vxReleaseImage(&state->mask);
    if (state->frame_delay) vxReleaseDelay(&state->frame_delay);
    if (state->renderer) delete state->renderer.get();
    if (state->source) delete state->source.get();
    if (state->tracker) delete state->tracker;
    if (state->context) delete state->context;
    memset(state, 0, sizeof(*state));
    free(state);
}

static int initFrameSource(BenchmarkState *state, std::string sourceUri)
{
    // Create a OVXIO-based frame source
    state->source = ovxio::createDefaultFrameSource(*(state->context), sourceUri);

    if (!state->source || !state->source->open())
    {
        std::cerr << "Error: Can't open source URI " << sourceUri << std::endl;
        return 0;
    }

    if (state->source->getSourceType() == ovxio::FrameSource::SINGLE_IMAGE_SOURCE)
    {
        std::cerr << "Error: Can't work on a single image." << std::endl;
        return 0;
    }
    return 1;
}

static int initTracker(BenchmarkState *state, std::string maskFile, nvx::FeatureTracker::Params ft_params)
{
    // Load optional mask image if needed. To be used later by tracker

    state->mask = NULL;

#if defined USE_OPENCV || defined USE_GSTREAMER
    if (!maskFile.empty())
    {
        state->mask = ovxio::loadImageFromFile(*(state->context), maskFile, VX_DF_IMAGE_U8);

        vx_uint32 mask_width = 0, mask_height = 0;
        NVXIO_SAFE_CALL( vxQueryImage(state->mask, VX_IMAGE_ATTRIBUTE_WIDTH, &mask_width, sizeof(mask_width)) );
        NVXIO_SAFE_CALL( vxQueryImage(state->mask, VX_IMAGE_ATTRIBUTE_HEIGHT, &mask_height, sizeof(mask_height)) );

        ovxio::FrameSource::Parameters sourceParams = state->source->getConfiguration();
        if (mask_width != sourceParams.frameWidth || mask_height != sourceParams.frameHeight)
        {
            std::cerr << "Error: The mask must have the same size as the input source." << std::endl;
            return 0;
        }
    }
#endif
    // Create FeatureTracker instance
    state->tracker = nvx::FeatureTracker::create(*(state->context), ft_params);

    ovxio::FrameSource::FrameStatus frameStatus;

    // The first frame is read to initialize the tracker (tracker->init())
    // and immediately "age" the delay. See the FeatureTrackerPyrLK::init()
    // call in the file feature_tracker.cpp for further details
    do
    {
        frameStatus = state->source->fetch(state->frame);
    } while (frameStatus == ovxio::FrameSource::TIMEOUT);

    if (frameStatus == ovxio::FrameSource::CLOSED)
    {
        std::cerr << "Error: Source has no frames" << std::endl;
        return 0;
    }

    state->tracker->init(state->frame, state->mask);

    vxAgeDelay(state->frame_delay);
    return 1;
}

static int initRender(BenchmarkState *state)
{
    if (!state->shouldRender)
    {
        return 1;
    }

    ovxio::FrameSource::Parameters sourceParams = state->source->getConfiguration();

    // Create a OVXIO-based render
    state->renderer = ovxio::createDefaultRender( *(state->context), "Feature Tracker Demo",
            sourceParams.frameWidth,
            sourceParams.frameHeight);

    if (!state->renderer)
    {
        std::cerr << "Error: Can't create a renderer" << std::endl;
        return 0;
    }

    state->eventData = EventData();
    state->renderer->setOnKeyboardEventCallback(eventCallback, &state->eventData);
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

    // The input video filename is read into sourceURI and the configuration
    // parameters are read into configFile
    std::string sourceUri = app.findSampleFilePath("cars.mp4");
    std::string configFile = app.findSampleFilePath("feature_tracker_demo_config.ini");

#if defined USE_OPENCV || defined USE_GSTREAMER
    std::string maskFile;
    // TODO: maskFile string should be set here with the value provided in
    // (Initializationparameters *)params.
#endif

    // For now, this call is only setting up graphics output stream.
    app.init(1, NULL);

    // Create OpenVX context
    state->context = new ovxio::ContextGuard;
    vxDirective(*(state->context), VX_DIRECTIVE_ENABLE_PERFORMANCE);

    // Messages generated by the OpenVX framework will be processed by
    // ovxio::stdoutLogCallback
    vxRegisterLogCallback(*(state->context), &ovxio::stdoutLogCallback, vx_false_e);

    // Read and check input parameters
    nvx::FeatureTracker::Params ft_params;
    std::string error;
    if (!read(configFile, ft_params, error))
    {
        std::cout<<error;
        Cleanup(state);
        return NULL;
    }

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

    ovxio::FrameSource::Parameters sourceParams = state->source->getConfiguration();
    // Create OpenVX Image to hold frames from video source
    vx_image frameExemplar = vxCreateImage(*(state->context),
            sourceParams.frameWidth, sourceParams.frameHeight, VX_DF_IMAGE_RGBX);
    NVXIO_CHECK_REFERENCE(frameExemplar);
    state->frame_delay = vxCreateDelay(*(state->context), (vx_reference)frameExemplar, 2);
    NVXIO_CHECK_REFERENCE(state->frame_delay);
    vxReleaseImage(&frameExemplar);

    // Frame delay object is created to hold previous and current frames
    // from video source in the state->frame_delay variable
    state->prevFrame = (vx_image)vxGetReferenceFromDelay(state->frame_delay, -1);
    state->frame = (vx_image)vxGetReferenceFromDelay(state->frame_delay, 0);

    if (!initTracker(state, maskFile, ft_params))
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

static int Execute(void *data)
{
    BenchmarkState *state = (BenchmarkState *)data;
    try
    {

        // Run the main processing loop in which we read subsequent frames and
        // then pass them to tracker->track() and then aging the delay again.
        // See the FeatureTracker::track() call in the feature_tracker.cpp for
        // details. The aging mechanism allows the algorithm to access current
        // and previous frame. The tracker gets the featureList from the prevoius
        // frame and the CurrentFrame and draws the arrows between them

        ovxio::FrameSource::FrameStatus frameStatus;
loop:
        // When it's not rendered, pause isn't an option. The frame is processed
        // as it goes.
        // When it's rendered, need to check whether it's paused.
        if (!state->shouldRender || (state->shouldRender && !state->eventData.pause))
        {
            nvx::Timer procTimer;
            frameStatus = state->source->fetch(state->frame);

            if (frameStatus == ovxio::FrameSource::TIMEOUT)
            {
                goto loop;
            }
            if (frameStatus == ovxio::FrameSource::CLOSED)
            {
                if (!state->source->open())
                {
                    std::cerr << "Error: Failed to reopen the source" << std::endl;
                    Cleanup(state);
                    return 0;
                }
                goto loop;
            }

            // Process
            state->tracker->track(state->frame, state->mask);
        }

        // Show the previous frame
        if (state->shouldRender && state->renderer)
        {
            state->renderer->putImage(state->prevFrame);

            // Draw arrows & state
            ovxio::Render::FeatureStyle featureStyle = { { 255, 0, 0, 255 }, 4.0f };
            ovxio::Render::LineStyle arrowStyle = {{0, 255, 0, 255}, 1};

            vx_array old_points = state->tracker->getPrevFeatures();
            vx_array new_points = state->tracker->getCurrFeatures();

            state->renderer->putArrows(old_points, new_points, arrowStyle);
            state->renderer->putFeatures(old_points, featureStyle);

            if (!state->renderer->flush())
            {
                state->eventData.shouldStop = true;
            }
        }
        if (!state->eventData.pause)
        {
            vxAgeDelay(state->frame_delay);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        Cleanup(state);
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
    return "VisionWorks demo: feature tracker";
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
