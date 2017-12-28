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

#include "NVX/Application.hpp"
#include "NVX/ConfigParser.hpp"
#include "OVX/FrameSourceOVX.hpp"
#include "OVX/RenderOVX.hpp"
#include "NVX/SyncTimer.hpp"
#include "OVX/UtilityOVX.hpp"

#include "iterative_motion_estimator.hpp"

// for cuda_scheduling_examiner
#include <library_interface.h>
#include "third_party/cJSON.h"

// Process events
struct EventData
{
    EventData() : stop(false), pause(false) {}

    bool stop;
    bool pause;
};

// Holds the local state for one instance of this benchmark.
typedef struct {
    IterativeMotionEstimator::Params me_params;
    ovxio::ContextGuard *context;
    std::unique_ptr<ovxio::FrameSource> frameSource;
    IterativeMotionEstimator *ime;
    EventData eventData;
    std::unique_ptr<ovxio::Render> renderer;
    vx_delay frame_delay;
    vx_image prevFrame;
    vx_image currFrame;
    bool shouldRender;
} BenchmarkState;

static void keyboardEventCallback(void* eventData, vx_char key, vx_uint32, vx_uint32)
{
    EventData* data = static_cast<EventData*>(eventData);

    if (key == 27) // escape
    {
        data->stop = true;
    }
    else if (key == ' ') // space
    {
        data->pause = !data->pause;
    }
}

// Parse configuration file
static bool read(const std::string& configFile,
        IterativeMotionEstimator::Params& params,
        std::string& message)
{
    std::unique_ptr<nvxio::ConfigParser> parser(nvxio::createConfigParser());

    parser->addParameter("biasWeight",
            nvxio::OptionHandler::real(&params.biasWeight,
                nvxio::ranges::atLeast(0.0f)));
    parser->addParameter("mvDivFactor",
            nvxio::OptionHandler::integer(&params.mvDivFactor,
                nvxio::ranges::atLeast(0)
                &
                nvxio::ranges::atMost(16)));
    parser->addParameter("smoothnessFactor",
            nvxio::OptionHandler::real(&params.smoothnessFactor,
                nvxio::ranges::atLeast(0.0f)));

    message = parser->parse(configFile);

    return message.empty();
}

static void Cleanup(void *data) {
    // Release all objects
    BenchmarkState *state = (BenchmarkState *)data;
    if (state->frame_delay) vxReleaseDelay(&state->frame_delay);
    if (state->renderer) delete state->renderer.get();
    if (state->frameSource) delete state->frameSource.get();
    if (state->ime) delete state->ime;
    if (state->context) delete state->context;
    memset(state, 0, sizeof(*state));
    free(state);
}

static int initFrameSource(BenchmarkState *state, std::string sourceUri)
{
    // Create a Frame Source
    state->frameSource = ovxio::createDefaultFrameSource(*(state->context), sourceUri);

    if (!state->frameSource || !state->frameSource->open())
    {
        std::cerr << "Error: cannot open frame source!" << std::endl;
        return 0;
    }

    if (state->frameSource->getSourceType() == ovxio::FrameSource::SINGLE_IMAGE_SOURCE)
    {
        std::cerr << "Can't work on a single image." << std::endl;
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

    // Create a Render
    state->renderer = ovxio::createDefaultRender(*(state->context), "Motion Estimation Demo",
            frameConfig.frameWidth, frameConfig.frameHeight);

    if (!state->renderer)
    {
        std::cerr << "Error: Cannot create state->renderer!" << std::endl;
        return 0;
    }

    state->renderer->setOnKeyboardEventCallback(keyboardEventCallback, &state->eventData);
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

static int initIME(BenchmarkState *state)
{
    // Create algorithm
    state->ime = new IterativeMotionEstimator(*(state->context));
    ovxio::FrameSource::FrameStatus frameStatus;
    do
    {
        frameStatus = state->frameSource->fetch(state->prevFrame);
    } while (frameStatus == ovxio::FrameSource::TIMEOUT);
    if (frameStatus == ovxio::FrameSource::CLOSED)
    {
        std::cerr << "Source has no frames" << std::endl;
        return 0;
    }
    state->ime->init(state->prevFrame, state->currFrame, state->me_params);
    return 1;
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
    std::string sourceUri = app.findSampleFilePath("pedestrians.mp4");
    std::string configFile = app.findSampleFilePath("motion_estimation_demo_config.ini");

    app.init(1, NULL);

    // Reads and checks input parameters
    state->me_params = IterativeMotionEstimator::Params();
    std::string error;
    if (!read(configFile, state->me_params, error))
    {
        std::cout << error;
        Cleanup(state);
        return NULL;
    }

    // Create OpenVX context
    state->context = new ovxio::ContextGuard;
    vxDirective(*(state->context), VX_DIRECTIVE_ENABLE_PERFORMANCE);

    // Messages generated by the OpenVX framework will be processed by ovxio::stdoutLogCallback
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

    ovxio::FrameSource::Parameters frameConfig = state->frameSource->getConfiguration();

    // Create OpenVX Image to hold frames from video source
    vx_image frameExemplar = vxCreateImage(*(state->context),
            frameConfig.frameWidth, frameConfig.frameHeight, VX_DF_IMAGE_RGBX);
    NVXIO_CHECK_REFERENCE(frameExemplar);
    state->frame_delay = vxCreateDelay(*(state->context), (vx_reference)frameExemplar, 2);
    NVXIO_CHECK_REFERENCE(state->frame_delay);
    vxReleaseImage(&frameExemplar);

    state->prevFrame = (vx_image)vxGetReferenceFromDelay(state->frame_delay, -1);
    state->currFrame = (vx_image)vxGetReferenceFromDelay(state->frame_delay, 0);

    if (!initIME(state))
    {
        Cleanup(state);
        return NULL;
    }

    return state;
}

static int CopyIn(void *data) {
    BenchmarkState *state = (BenchmarkState *)data;
    ovxio::FrameSource::FrameStatus frameStatus;
    if (!state->shouldRender || (state->shouldRender && !state->eventData.pause))
    {
        // Grab next frame
        frameStatus = state->frameSource->fetch(state->currFrame);

        if (frameStatus == ovxio::FrameSource::TIMEOUT)
        {
            std::cerr << "Error: frame source featch TIMEOUT" << std::endl;
            return 0;
        }

        if (frameStatus == ovxio::FrameSource::CLOSED)
        {
            if (!state->frameSource->open())
            {
                std::cerr << "Failed to reopen the source" << std::endl;
                return 0;
            }

            do
            {
                frameStatus = state->frameSource->fetch(state->prevFrame);
            } while (frameStatus == ovxio::FrameSource::TIMEOUT);

            if (frameStatus == ovxio::FrameSource::CLOSED)
            {
                std::cerr << "Source has no frames" << std::endl;
                return 0;
            }

            state->ime->init(state->prevFrame, state->currFrame, state->me_params);
            return 1;
        }
    }
    return 1;
}

static int Execute(void *data)
{
    try
    {
        BenchmarkState *state = (BenchmarkState *)data;

        // When it's not rendered, pause isn't an option. The frame is processed
        // as it goes.
        // When it's rendered, need to check whether it's paused.
        if (!state->shouldRender || (state->shouldRender && !state->eventData.pause))
        {
            // Process
            state->ime->process();
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
    // state->renderer
    if (state->shouldRender && state->renderer)
    {
        state->renderer->putImage(state->prevFrame);

        ovxio::Render::MotionFieldStyle mfStyle = {
            {  0u, 255u, 255u, 255u} // color
        };

        state->renderer->putMotionField(state->ime->getMotionField(), mfStyle);

        if (!state->renderer->flush())
        {
            state->eventData.stop = true;
        }
    }
    if (!state->eventData.pause)
    {
        vxAgeDelay(state->frame_delay);
    }
    return 1;
}

static const char* GetName(void) {
    return "VisionWorks demo: motion estimation";
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
