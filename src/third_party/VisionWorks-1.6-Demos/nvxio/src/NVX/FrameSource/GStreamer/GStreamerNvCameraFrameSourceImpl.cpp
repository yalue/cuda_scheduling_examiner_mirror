/*
# Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
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

#ifdef USE_NVGSTCAMERA

#include "GStreamerNvCameraFrameSourceImpl.hpp"

#include <gst/app/gstappsink.h>
#include <sstream>


namespace nvidiaio
{

struct NvCameraConfigs
{
    vx_uint32 frameWidth, frameHeight, fps;
};

static const NvCameraConfigs configs[4] =
{
    { vx_uint32(2592), vx_uint32(1944), vx_uint32(30)  }, // 0
    { vx_uint32(2592), vx_uint32(1458), vx_uint32(30)  }, // 1
    { vx_uint32(1280), vx_uint32(720) , vx_uint32(120) }, // 2
    { vx_uint32(2592), vx_uint32(1944), vx_uint32(24)  }  // 3
};

GStreamerNvCameraFrameSourceImpl::GStreamerNvCameraFrameSourceImpl(uint cameraIdx_) :
    GStreamerEGLStreamSinkFrameSourceImpl(nvxio::FrameSource::CAMERA_SOURCE, "GStreamerNvCameraFrameSource", false),
    cameraIdx(cameraIdx_)
{
}

bool GStreamerNvCameraFrameSourceImpl::setConfiguration(const FrameSource::Parameters& params)
{
    NVXIO_ASSERT(end);

    configuration.frameHeight = params.frameHeight;
    configuration.frameWidth = params.frameWidth;
    configuration.fps = params.fps;

    NVXIO_ASSERT((params.format == NVXCU_DF_IMAGE_NV12) ||
                 (params.format == NVXCU_DF_IMAGE_U8) ||
                 (params.format == NVXCU_DF_IMAGE_RGB) ||
                 (params.format == NVXCU_DF_IMAGE_RGBX)||
                 (params.format == NVXCU_DF_IMAGE_NONE));


    configuration.format = params.format;

    return true;
}

bool GStreamerNvCameraFrameSourceImpl::InitializeGstPipeLine()
{
    // select config with max FPS value to be default
    NvCameraConfigs nvcameraconfig = configs[2];

    // use user specified camera config
    if ( (configuration.frameWidth != (vx_uint32)-1) &&
         (configuration.frameHeight != (vx_uint32)-1) )
    {
        nvcameraconfig.frameWidth = configuration.frameWidth;
        nvcameraconfig.frameHeight = configuration.frameHeight;
        nvcameraconfig.fps = 30;

        // select FPS default for the specified config
        for (vx_size i = 0; i < ovxio::dimOf(configs); ++i)
        {
            if ((nvcameraconfig.frameWidth == configs[i].frameWidth) &&
                (nvcameraconfig.frameHeight == configs[i].frameHeight))
            {
                nvcameraconfig.fps = configs[i].fps;
                break;
            }
        }
    }

    if (configuration.fps == (vx_uint32)-1)
        configuration.fps = nvcameraconfig.fps;

    end = true;

    pipeline = GST_PIPELINE(gst_pipeline_new(nullptr));
    if (!pipeline)
    {
        NVXIO_PRINT("Cannot create Gstreamer pipeline");
        return false;
    }

    bus = gst_pipeline_get_bus(GST_PIPELINE (pipeline));

    // create nvcamerasrc
    GstElement * nvcamerasrc = gst_element_factory_make("nvcamerasrc", nullptr);
    if (!nvcamerasrc)
    {
        NVXIO_PRINT("Cannot create nvcamerasrc");
        NVXIO_PRINT("\"nvcamerasrc\" element is not available on this platform");
        FinalizeGstPipeLine();

        return false;
    }

    std::ostringstream stream;
    stream << configuration.fps << " " << configuration.fps;
    std::string fpsRange = stream.str();

    g_object_set(G_OBJECT(nvcamerasrc),
                 "sensor-id", cameraIdx,
                 "fpsRange", fpsRange.c_str(),
                 nullptr);

    gst_bin_add(GST_BIN(pipeline), nvcamerasrc);

    // create nvvideosink element
    GstElement * nvvideosink = gst_element_factory_make("nvvideosink", nullptr);
    if (!nvvideosink)
    {
        NVXIO_PRINT("Cannot create nvvideosink element");
        FinalizeGstPipeLine();

        return false;
    }

    g_object_set(G_OBJECT(nvvideosink),
                 "display", context.display,
                 "stream", context.stream,
                 "fifo", fifoMode,
                 "max-lateness", G_GINT64_CONSTANT(-1),
                 "throttle-time", G_GUINT64_CONSTANT(0),
                 "render-delay", G_GUINT64_CONSTANT(0),
                 "qos", FALSE,
                 "sync", FALSE,
                 "async", TRUE,
                 nullptr);

    gst_bin_add(GST_BIN(pipeline), nvvideosink);

    // link elements
    stream.str(std::string());
    stream << "video/x-raw(memory:NVMM), width=(int)" << nvcameraconfig.frameWidth << ", "
              "height=(int)" << nvcameraconfig.frameHeight << ", format=(string){I420}, "
              "framerate=(fraction)" << nvcameraconfig.fps << "/1;";

    std::unique_ptr<GstCaps, GStreamerObjectDeleter> caps_nvvidconv(
        gst_caps_from_string(stream.str().c_str()));

    if (!caps_nvvidconv)
    {
        NVXIO_PRINT("Failed to create caps");
        FinalizeGstPipeLine();

        return false;
    }

    if (!gst_element_link_filtered(nvcamerasrc, nvvideosink, caps_nvvidconv.get()))
    {
        NVXIO_PRINT("GStreamer: cannot link nvcamerasrc -> nvvideosink using caps");
        FinalizeGstPipeLine();

        return false;
    }

    // Force pipeline to play video as fast as possible, ignoring system clock
    gst_pipeline_use_clock(pipeline, nullptr);

    GstStateChangeReturn status = gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_PLAYING);
    handleGStreamerMessages();

    if (status == GST_STATE_CHANGE_ASYNC)
    {
        // wait for status update
        status = gst_element_get_state(GST_ELEMENT(pipeline), nullptr, nullptr, GST_CLOCK_TIME_NONE);
    }
    if (status == GST_STATE_CHANGE_FAILURE)
    {
        NVXIO_PRINT("GStreamer: unable to start playback");
        FinalizeGstPipeLine();

        return false;
    }

    vx_uint32 initialFPS = configuration.fps;

    if (!updateConfiguration(nvvideosink, nvcamerasrc, configuration))
    {
        FinalizeGstPipeLine();
        return false;
    }

    // if initialFPS is specified, we should use this, because
    // retrieved via the updateConfiguration function FPS corresponds
    // to camera config FPS
    if (initialFPS != (vx_uint32)-1)
        configuration.fps = initialFPS;

    end = false;

    return true;
}

} // namespace nvidiaio

#endif // USE_NVGSTCAMERA
