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

#ifdef USE_GSTREAMER

#include "FrameSource/GStreamer/GStreamerCameraFrameSourceImpl.hpp"

#include <gst/app/gstappsink.h>

#include <sstream>

namespace nvidiaio
{

GStreamerCameraFrameSourceImpl::GStreamerCameraFrameSourceImpl(uint cameraIdx_) :
    GStreamerBaseFrameSourceImpl(nvxio::FrameSource::CAMERA_SOURCE, "GstreamerCameraFrameSource"),
    cameraIdx(cameraIdx_)
{
}

bool GStreamerCameraFrameSourceImpl::setConfiguration(const Parameters& params)
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


bool GStreamerCameraFrameSourceImpl::InitializeGstPipeLine()
{
    GstStateChangeReturn status;
    end = true;

    pipeline = GST_PIPELINE(gst_pipeline_new(nullptr));
    if (!pipeline)
    {
        NVXIO_PRINT("Cannot create Gstreamer pipeline");
        return false;
    }

    bus = gst_pipeline_get_bus(GST_PIPELINE (pipeline));

    // create v4l2src
    GstElement * v4l2src = gst_element_factory_make("v4l2src", nullptr);
    if (!v4l2src)
    {
        NVXIO_PRINT("Cannot create v4l2src");
        FinalizeGstPipeLine();

        return false;
    }

    std::ostringstream cameraDev;
    cameraDev << "/dev/video" << cameraIdx;
    g_object_set(G_OBJECT(v4l2src), "device", cameraDev.str().c_str(), nullptr);

    gst_bin_add(GST_BIN(pipeline), v4l2src);

    // create color convert element
    GstElement * color = gst_element_factory_make(COLOR_ELEM, nullptr);
    if (!color)
    {
        NVXIO_PRINT("Cannot create %s element", COLOR_ELEM);
        FinalizeGstPipeLine();

        return false;
    }

    gst_bin_add(GST_BIN(pipeline), color);

    // create appsink element
    sink = gst_element_factory_make("appsink", nullptr);
    if (!sink)
    {
        NVXIO_PRINT("Cannot create appsink element");
        FinalizeGstPipeLine();

        return false;
    }

#if FULL_GST_VERSION >= VERSION_NUM(1,7,2)
    g_object_set(GST_ELEMENT(sink), "wait-on-eos", FALSE, nullptr);
#endif

    gst_bin_add(GST_BIN(pipeline), sink);

    // if initial values for FrameSource::Parameters are not
    // specified, let's set them manually to prevent very huge images
    if (configuration.frameWidth == (vx_uint32)-1)
        configuration.frameWidth = 1920;
    if (configuration.frameHeight == (vx_uint32)-1)
        configuration.frameHeight = 1080;
    if (configuration.fps == (vx_uint32)-1)
        configuration.fps = 30;

#if GST_VERSION_MAJOR == 0
    std::unique_ptr<GstCaps, GStreamerObjectDeleter> caps_v42lsrc(
        gst_caps_new_simple ("video/x-raw-rgb",
                             "width", GST_TYPE_INT_RANGE, 1, (int)configuration.frameWidth,
                             "height", GST_TYPE_INT_RANGE, 1, (int)configuration.frameHeight,
                             "framerate", GST_TYPE_FRACTION, (int)configuration.fps,
                             nullptr));
#else
    std::ostringstream stream;
    stream << "video/x-raw, format=(string){RGB}, width=[1," << configuration.frameWidth <<
              "], height=[1," << configuration.frameHeight << "], framerate=" << configuration.fps << "/1;";

    std::unique_ptr<GstCaps, GStreamerObjectDeleter> caps_v42lsrc(gst_caps_from_string(stream.str().c_str()));
#endif

    if (!caps_v42lsrc)
    {
        NVXIO_PRINT("Failed to create caps");
        FinalizeGstPipeLine();

        return false;
    }

    // link elements
    if (!gst_element_link_filtered(v4l2src, color, caps_v42lsrc.get()))
    {
        NVXIO_PRINT("GStreamer: cannot link v4l2src -> color using caps");
        FinalizeGstPipeLine();

        return false;
    }

    // link elements
    if (!gst_element_link(color, sink))
    {
        NVXIO_PRINT("GStreamer: cannot link color -> appsink");
        FinalizeGstPipeLine();

        return false;
    }

    gst_app_sink_set_max_buffers (GST_APP_SINK(sink), 1);
    gst_app_sink_set_drop (GST_APP_SINK(sink), true);

    // do not emit signals: all calls will be synchronous and blocking
    gst_app_sink_set_emit_signals (GST_APP_SINK(sink), 0);

#if GST_VERSION_MAJOR == 0
    std::unique_ptr<GstCaps, GStreamerObjectDeleter> caps_appsink;

    if (configuration.format == NVXCU_DF_IMAGE_U8)
    {
        caps_appsink.reset(gst_caps_new_simple("video/x-raw-gray",
                                               "bpp",        G_TYPE_INT, 8,
                                               nullptr));
    }
    else if (configuration.format == NVXCU_DF_IMAGE_RGB)
    {
        caps_appsink.reset(gst_caps_new_simple("video/x-raw-rgb",
                                               "bpp",        G_TYPE_INT, 24,
                                               "red_mask",   G_TYPE_INT, 0xFF0000,
                                               "green_mask", G_TYPE_INT, 0x00FF00,
                                               "blue_mask",  G_TYPE_INT, 0x0000FF,
                                               nullptr));
    }
    else if (configuration.format == NVXCU_DF_IMAGE_RGBX ||
             configuration.format == NVXCU_DF_IMAGE_NONE)
    {
        caps_appsink.reset(gst_caps_new_simple("video/x-raw-rgb",
                                               "depth",      G_TYPE_INT, 32,
                                               "bpp",        G_TYPE_INT, 32,
                                               "endianness", G_TYPE_INT, 4321,
                                               "red_mask",   G_TYPE_INT, 0xFF000000,
                                               "green_mask", G_TYPE_INT, 0x00FF0000,
                                               "blue_mask",  G_TYPE_INT, 0x0000FF00,
                                               "alpha_mask", G_TYPE_INT, 0x000000FF,
                                               nullptr));
    }
    else if (configuration.format == NVXCU_DF_IMAGE_NV12)
    {
        caps_appsink.reset(gst_caps_new_simple("video/x-raw-yuv",
                                               "format", GST_TYPE_FOURCC, GST_MAKE_FOURCC ('N', 'V', '1', '2'),
                                               nullptr));
    }
    else
        NVXIO_THROW_EXCEPTION("Unsupported image format");
#else
    std::string caps_string("video/x-raw, format=(string){");
    if (configuration.format == NVXCU_DF_IMAGE_U8)
        caps_string += "GRAY8";
    else if (configuration.format == NVXCU_DF_IMAGE_RGB)
        caps_string += "RGB";
    else if (configuration.format == NVXCU_DF_IMAGE_RGBX ||
             configuration.format == NVXCU_DF_IMAGE_NONE)
        caps_string += "RGBA";
    else if (configuration.format == NVXCU_DF_IMAGE_NV12)
        caps_string += "NV12";
    else
        NVXIO_THROW_EXCEPTION("Unsupported image format");

    caps_string += "};";

    std::unique_ptr<GstCaps, GStreamerObjectDeleter> caps_appsink(
            gst_caps_from_string(caps_string.c_str()));
#endif
    gst_app_sink_set_caps(GST_APP_SINK(sink), caps_appsink.get());

    // Force pipeline to play video as fast as possible, ignoring system clock
    gst_pipeline_use_clock(pipeline, nullptr);

    status = gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_PLAYING);
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

    // GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline");

    // explicitly set params to -1 to ensure
    // their update in the updateConfiguration() function
    configuration.frameWidth = (uint32_t)-1;
    configuration.frameHeight = (uint32_t)-1;
    configuration.fps = (uint32_t)-1;

    if (!updateConfiguration(sink, color, configuration))
    {
        FinalizeGstPipeLine();
        return false;
    }

    end = false;

    return true;
}

} // namespace nvidiaio

#endif // USE_GSTREAMER
