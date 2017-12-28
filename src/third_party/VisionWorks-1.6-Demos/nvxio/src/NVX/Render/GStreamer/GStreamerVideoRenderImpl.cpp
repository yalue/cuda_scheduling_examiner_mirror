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

#if defined USE_GUI && defined USE_GSTREAMER

#include "Render/GStreamer/GStreamerVideoRenderImpl.hpp"
#include "Private/GStreamerUtils.hpp"

namespace {

int getVideoBitrate()
{
    if (const char * const fromEnv = ::getenv("NVXIO_VIDEO_RENDER_BITRATE")) try
    {
        return std::max(std::stoi(fromEnv), 0);
    }
    catch (...)
    {
        return 0;
    }

    return -1;
}

}

nvidiaio::GStreamerVideoRenderImpl::GStreamerVideoRenderImpl() :
    GStreamerBaseRenderImpl(nvxio::Render::VIDEO_RENDER, "GStreamerVideoOpenGlRender")
{
}

bool nvidiaio::GStreamerVideoRenderImpl::InitializeGStreamerPipeline()
{
    std::ostringstream stream;

    pipeline = GST_PIPELINE(gst_pipeline_new(nullptr));
    if (!pipeline)
    {
        NVXIO_PRINT("Cannot create Gstreamer pipeline");
        return false;
    }

    bus = gst_pipeline_get_bus(GST_PIPELINE (pipeline));

    // create appsrc
    GstElement * appsrcelem = gst_element_factory_make("appsrc", nullptr);
    if (!appsrcelem)
    {
        NVXIO_PRINT("Cannot create appsrc");
        FinalizeGStreamerPipeline();

        return false;
    }

    g_object_set(G_OBJECT(appsrcelem),
                 "is-live", 0,
                 "num-buffers", -1,
                 "emit-signals", 0,
                 "block", 1,
                 "size", static_cast<guint64>(wndHeight_ * wndWidth_ * 4),
                 "format", GST_FORMAT_TIME,
                 "stream-type", GST_APP_STREAM_TYPE_STREAM,
                 nullptr);

    appsrc = GST_APP_SRC_CAST(appsrcelem);
#if GST_VERSION_MAJOR == 0
    std::unique_ptr<GstCaps, nvidiaio::GStreamerObjectDeleter> caps(
                gst_caps_new_simple("video/x-raw-rgb",
                                    "bpp", G_TYPE_INT, 32,
                                    "endianness", G_TYPE_INT, 4321,
                                    "red_mask", G_TYPE_INT, -16777216,
                                    "green_mask", G_TYPE_INT, 16711680,
                                    "blue_mask", G_TYPE_INT, 65280,
                                    "alpha_mask", G_TYPE_INT, 255,
                                    "width", G_TYPE_INT, wndWidth_,
                                    "height", G_TYPE_INT, wndHeight_,
                                    "framerate", GST_TYPE_FRACTION, GSTREAMER_DEFAULT_FPS, 1,
                                    nullptr));
    if (!caps)
    {
        NVXIO_PRINT("Failed to create caps");
        FinalizeGStreamerPipeline();

        return false;
    }

#else
    // support 4 channel 8 bit data
    stream << "video/x-raw"
           << ", width=" << wndWidth_
           << ", height=" << wndHeight_
           << ", format=(string){RGBA}"
           << ", framerate=" << GSTREAMER_DEFAULT_FPS << "/1;";
    std::unique_ptr<GstCaps, nvidiaio::GStreamerObjectDeleter> caps(
                gst_caps_from_string(stream.str().c_str()));

    if (!caps)
    {
        NVXIO_PRINT("Failed to create caps");
        FinalizeGStreamerPipeline();

        return false;
    }

    gst_caps_ref(caps.get());
    caps.reset(gst_caps_fixate(caps.get()));
#endif

    gst_app_src_set_caps(appsrc, caps.get());

    gst_bin_add(GST_BIN(pipeline), appsrcelem);

    // create color convert element
    GstElement * color = gst_element_factory_make(COLOR_ELEM, nullptr);
    if (!color)
    {
        NVXIO_PRINT("Cannot create " COLOR_ELEM " element");
        FinalizeGStreamerPipeline();

        return false;
    }
    gst_bin_add(GST_BIN(pipeline), color);

    // create videoflip element
    GstElement * videoflip = gst_element_factory_make("videoflip", nullptr);
    if (!videoflip)
    {
        NVXIO_PRINT("Cannot create videoflip element");
        FinalizeGStreamerPipeline();

        return false;
    }

    g_object_set(G_OBJECT(videoflip), "method", 5, nullptr);

    gst_bin_add(GST_BIN(pipeline), videoflip);

    // create encodelem element
    GstElement * encodelem = gst_element_factory_make(ENCODE_ELEM, nullptr);
    if (!encodelem)
    {
        NVXIO_PRINT("Cannot create " ENCODE_ELEM " element");
        FinalizeGStreamerPipeline();

        return false;
    }

    const int bitrate = getVideoBitrate();
    if (bitrate > 0)
    {
        g_object_set(G_OBJECT(encodelem), "bitrate", bitrate, nullptr);
    }
    else if (bitrate == 0)
    {
        NVXIO_PRINT("Incorrect target video bitrate");
        FinalizeGStreamerPipeline();

        return false;
    }


    gst_bin_add(GST_BIN(pipeline), encodelem);

    // create avimux element
    GstElement * avimux = gst_element_factory_make("avimux", nullptr);
    if (!avimux)
    {
        NVXIO_PRINT("Cannot create avimux element");
        FinalizeGStreamerPipeline();

        return false;
    }

    gst_bin_add(GST_BIN(pipeline), avimux);

    // create filesink element
    GstElement * filesink = gst_element_factory_make("filesink", nullptr);
    if (!filesink)
    {
        NVXIO_PRINT("Cannot create filesink element");
        FinalizeGStreamerPipeline();

        return false;
    }

    g_object_set(G_OBJECT(filesink), "location", windowTitle_.c_str(),
                                     "append", 0, nullptr);

    gst_bin_add(GST_BIN(pipeline), filesink);


    // link elements
    if (!gst_element_link_many(appsrcelem, color, videoflip,
                               encodelem, avimux, filesink, nullptr))
    {
        NVXIO_PRINT("GStreamer: cannot link appsrc -> " COLOR_ELEM
                    " -> videoflip -> " ENCODE_ELEM " -> avimux -> filesink");
        FinalizeGStreamerPipeline();

        return false;
    }

    // Force pipeline to play video as fast as possible, ignoring system clock
    gst_pipeline_use_clock(pipeline, nullptr);

    num_frames = 0;

    GstStateChangeReturn status = gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_PLAYING);
    if (status == GST_STATE_CHANGE_FAILURE)
    {
        NVXIO_PRINT("GStreamer: unable to start playback");
        FinalizeGStreamerPipeline();

        return false;
    }

    return true;
}

#endif // USE_GUI && USE_GSTREAMER
