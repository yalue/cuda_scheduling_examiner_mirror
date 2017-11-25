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

#include <NVX/Application.hpp>
#include <OVX/FrameSourceOVX.hpp>
#include <OVX/RenderOVX.hpp>
#include <NVX/SyncTimer.hpp>
#include <OVX/UtilityOVX.hpp>

#include "stabilizer.hpp"

// for cuda_scheduling_examiner
#include <library_interface.h>
#include "third_party/cJSON.h"

struct EventData
{
    EventData(): shouldStop(false), pause(false) {}
    bool shouldStop;
    bool pause;
};

// Holds the local state for one instance of this benchmark.
typedef struct {
    vx_image demoImg;
    vx_image leftRoi;
    vx_image rightRoi;
    vx_image frame;
    vx_image lastFrame;
    vx_delay orig_frame_delay;
    ovxio::ContextGuard *context;
    std::unique_ptr<ovxio::FrameSource> source;
    std::unique_ptr<ovxio::Render> renderer;
    EventData eventData;
    nvx::VideoStabilizer *stabilizer;
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
                         double proc_ms, double total_ms, float cropMargin)
{
    vx_uint32 renderWidth = renderer->getViewportWidth();

    std::ostringstream txt;
    txt << std::fixed << std::setprecision(1);

    const vx_int32 borderSize = 10;
    ovxio::Render::TextBoxStyle style = {{255, 255, 255, 255}, {0, 0, 0, 127},
        {renderWidth / 2 + borderSize, borderSize}};

    txt << "Source size: " << sourceParams.frameWidth << 'x' << sourceParams.frameHeight << std::endl;
    txt << "Algorithm: " << proc_ms << " ms / " << 1000.0 / proc_ms << " FPS" << std::endl;
    txt << "Display: " << total_ms  << " ms / " << 1000.0 / total_ms << " FPS" << std::endl;

    txt << std::setprecision(6);
    txt.unsetf(std::ios_base::floatfield);
    txt << "LIMITED TO " << nvxio::Application::get().getFPSLimit() << " FPS FOR DISPLAY" << std::endl;
    txt << "Space - pause/resume" << std::endl;
    txt << "Esc - close the demo";
    renderer->putTextViewport(txt.str(), style);

    const vx_int32 stabilizedLabelLenght = 100;
    style.origin.x = renderWidth - stabilizedLabelLenght;
    style.origin.y = borderSize;
    renderer->putTextViewport("stabilized", style);

    style.origin.x = renderWidth / 2 - stabilizedLabelLenght;
    renderer->putTextViewport("original", style);

    if (cropMargin > 0)
    {
        vx_uint32 dx = static_cast<vx_uint32>(cropMargin * sourceParams.frameWidth);
        vx_uint32 dy = static_cast<vx_uint32>(cropMargin * sourceParams.frameHeight);
        vx_rectangle_t rect = {dx, dy, sourceParams.frameWidth - dx, sourceParams.frameHeight - dy};

        ovxio::Render::DetectedObjectStyle rectStyle = {{""}, {255, 255, 255, 255}, 2, 0, false};
        renderer->putObjectLocation(rect, rectStyle);
    }
}

static void Cleanup(void *data) {
    BenchmarkState *state = (BenchmarkState *)data;
    if (state->orig_frame_delay) vxReleaseDelay(&state->orig_frame_delay);
    if (state->renderer)
    {
        state->renderer->close();
        delete state->renderer.get();
    }
    if (state->source) delete state->source.get();
    if (state->demoImg) vxReleaseImage(&state->demoImg);
    if (state->leftRoi) vxReleaseImage(&state->leftRoi);
    if (state->rightRoi) vxReleaseImage(&state->rightRoi);
    if (state->context) delete state->context;
    if (state->stabilizer) delete state->stabilizer;
    memset(state, 0, sizeof(*state));
    free(state);
}

static int initRender(BenchmarkState *state, vx_int32 demoImgHeight, vx_int32 demoImgWidth)
{
    if (!state->shouldRender)
    {
        return 1;
    }
    ovxio::FrameSource::Parameters sourceParams = state->source->getConfiguration();

    state->renderer = ovxio::createDefaultRender(*(state->context), "Video Stabilization Demo", demoImgWidth, demoImgHeight);

    if (!state->renderer)
    {
        std::cerr << "Error: Can't create a state->renderer" << std::endl;
        return 0;
    }

    state->eventData = EventData();
    state->renderer->setOnKeyboardEventCallback(eventCallback, &state->eventData);
    return 1;
}

static int initGraph(BenchmarkState *state, vx_int32 demoImgHeight, vx_int32 demoImgWidth,
        unsigned numOfSmoothingFrames, float cropMargin)
{
    // Create VideoStabilizer instance
    nvx::VideoStabilizer::VideoStabilizerParams vs_params;
    vs_params.numOfSmoothingFrames_ = numOfSmoothingFrames;
    vs_params.cropMargin_ = cropMargin;
    state->stabilizer = nvx::VideoStabilizer::createImageBasedVStab(*(state->context), vs_params);

    ovxio::FrameSource::FrameStatus frameStatus;

    do
    {
        frameStatus = state->source->fetch(state->frame);
    } while (frameStatus == ovxio::FrameSource::TIMEOUT);

    if (frameStatus == ovxio::FrameSource::CLOSED)
    {
        std::cerr << "Error: Source has no frames" << std::endl;
        return 0;
    }

    state->stabilizer->init(state->frame);

    vx_rectangle_t leftRect;
    NVXIO_SAFE_CALL( vxGetValidRegionImage(state->frame, &leftRect) );

    vx_rectangle_t rightRect;
    rightRect.start_x = leftRect.end_x;
    rightRect.start_y = leftRect.start_y;
    rightRect.end_x = 2 * leftRect.end_x;
    rightRect.end_y = leftRect.end_y;

    state->leftRoi = vxCreateImageFromROI(state->demoImg, &leftRect);
    NVXIO_CHECK_REFERENCE(state->leftRoi);
    state->rightRoi = vxCreateImageFromROI(state->demoImg, &rightRect);
    NVXIO_CHECK_REFERENCE(state->rightRoi);
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

static void* Initialize(InitializationParameters *params) {
    BenchmarkState *state = NULL;
    state = (BenchmarkState *) malloc(sizeof(*state));
    if (!state) return NULL;
    memset(state, 0, sizeof(*state));

    if (!processConfig(state, params->additional_info))
    {
        Cleanup(state);
        return NULL;
    }
    nvxio::Application &app =  nvxio::Application::get();

    // Parse command line arguments
    std::string videoFilePath = app.findSampleFilePath("parking.avi");
    unsigned numOfSmoothingFrames = 5;
    float cropMargin = 0.07f;

    app.init(1, NULL);

    // Create OpenVX context
    state->context = new ovxio::ContextGuard;
    vxRegisterLogCallback(*(state->context), &ovxio::stdoutLogCallback, vx_false_e);
    vxDirective(*(state->context), VX_DIRECTIVE_ENABLE_PERFORMANCE);

    // Create FrameSource
    state->source = ovxio::createDefaultFrameSource(*(state->context), videoFilePath);

    if (!state->source || !state->source->open())
    {
        std::cerr << "Error: Can't open state->source file: " << videoFilePath << std::endl;
        Cleanup(state);
        return NULL;
    }

    if (state->source->getSourceType() == ovxio::FrameSource::SINGLE_IMAGE_SOURCE)
    {
        std::cerr << "Error: Can't work on a single image." << std::endl;
        Cleanup(state);
        return NULL;
    }

    ovxio::FrameSource::Parameters sourceParams = state->source->getConfiguration();
    vx_int32 demoImgWidth = 2 * sourceParams.frameWidth;
    vx_int32 demoImgHeight = sourceParams.frameHeight;

    if (!initRender(state, demoImgHeight, demoImgWidth))
    {
        Cleanup(state);
        return NULL;
    }

    // Create OpenVX Image to hold frames from video source
    state->demoImg = vxCreateImage(*(state->context), demoImgWidth, demoImgHeight, VX_DF_IMAGE_RGBX);
    NVXIO_CHECK_REFERENCE(state->demoImg);

    vx_image frameExemplar = vxCreateImage(*(state->context),
            sourceParams.frameWidth, sourceParams.frameHeight,
            VX_DF_IMAGE_RGBX);
    vx_size orig_frame_delay_size = numOfSmoothingFrames + 2; //must have such size to be synchronized with the stabilized frames
    state->orig_frame_delay = vxCreateDelay(*(state->context),
            (vx_reference)frameExemplar, orig_frame_delay_size);
    NVXIO_CHECK_REFERENCE(state->orig_frame_delay);
    NVXIO_SAFE_CALL( nvx::initDelayOfImages(*(state->context), state->orig_frame_delay) );
    NVXIO_SAFE_CALL(vxReleaseImage(&frameExemplar));

    state->frame = (vx_image)vxGetReferenceFromDelay(state->orig_frame_delay, 0);
    state->lastFrame =
        (vx_image)vxGetReferenceFromDelay(state->orig_frame_delay, 1 -
                static_cast<vx_int32>(orig_frame_delay_size));

    if (!initGraph(state, demoImgHeight, demoImgWidth, numOfSmoothingFrames,
                cropMargin))
    {
        std::cerr << "Graph init failed" << std::endl;
        Cleanup(state);
        return NULL;
    }
    return state;
}

static int CopyIn(void *data) {
    return 1;
}

// main - Application entry point
//int main(int argc, char* argv[])
static int Execute(void *data)
{
    try
    {
        BenchmarkState *state = (BenchmarkState *)data;
        // Run processing loop
        // When it's not rendered, pause isn't an option. The frame is processed
        // as it goes.
        // When it's rendered, need to check whether it's paused.
        if (!state->shouldRender || (state->shouldRender && !state->eventData.pause))
        {
            // Process
            state->stabilizer->process(state->frame);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 0;
    }

    return 1;
}

static int CopyOut(void *data, TimingInformation *times) {
    BenchmarkState *state = (BenchmarkState *)data;
    times->kernel_count = 0;
    ovxio::FrameSource::FrameStatus frameStatus;
    if (!state->shouldRender || (state->shouldRender && !state->eventData.pause))
    {
        // Process
        state->stabilizer->process(state->frame);

        NVXIO_SAFE_CALL( vxAgeDelay(state->orig_frame_delay) );

        vx_image stabImg = state->stabilizer->getStabilizedFrame();
        NVXIO_SAFE_CALL( nvxuCopyImage(*(state->context), stabImg, state->rightRoi) );
        NVXIO_SAFE_CALL( nvxuCopyImage(*(state->context), state->lastFrame, state->leftRoi) );

        // Read frame
        frameStatus = state->source->fetch(state->frame);

        if (frameStatus == ovxio::FrameSource::TIMEOUT)
            return 1;
        else if (frameStatus == ovxio::FrameSource::CLOSED)
        {
            if (!state->source->open())
            {
                std::cerr << "Error: Failed to reopen the state->source" << std::endl;
                return 0;
            }
        }
    }

    if (state->shouldRender && state->renderer)
    {
        state->renderer->putImage(state->demoImg);

        if (!state->renderer->flush())
        {
            state->eventData.shouldStop = true;
        }
    }
    return 1;
}

static const char* GetName(void) {
    return "VisionWorks demo: video stabilizer";
}

// This should be the only function we export from the library, to provide
// pointers to all of the other functions.
int RegisterFunctions(BenchmarkLibraryFunctions *functions) {
    functions->initialize = Initialize;
    functions->copy_in = CopyIn;
    functions->execute = Execute;
    functions->copy_out = CopyOut;
    functions->cleanup = Cleanup;
    functions->get_name = GetName;
    return 1;
}
