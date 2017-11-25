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

#include "Render/GStreamer/GStreamerImagesRenderImpl.hpp"
#include "Private/GStreamerUtils.hpp"

nvidiaio::GStreamerImagesRenderImpl::GStreamerImagesRenderImpl() :
    GStreamerBaseRenderImpl(nvxio::Render::IMAGE_RENDER, "GStreamerImagesOpenGlRender")
{
}

bool nvidiaio::GStreamerImagesRenderImpl::InitializeGStreamerPipeline()
{
    // multifilesink does not report erros in case of URI is a directory,
    // let's make some check manually
    {
        // if uri is directory
        if (g_file_test(windowTitle_.c_str(), G_FILE_TEST_IS_DIR))
            return false;
    }

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
                 "is-live", FALSE,
                 "num-buffers", -1,
                 "emit-signals", FALSE,
                 "block", TRUE,
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

    // create color2 convert element
    GstElement * color2 = gst_element_factory_make(COLOR_ELEM, nullptr);
    if (!color2)
    {
        NVXIO_PRINT("Cannot create " COLOR_ELEM " element");
        FinalizeGStreamerPipeline();

        return false;
    }
    gst_bin_add(GST_BIN(pipeline), color2);

    // create pngenc element
    GstElement * pngenc = gst_element_factory_make("pngenc", nullptr);
    if (!pngenc)
    {
        NVXIO_PRINT("Cannot create pngenc element");
        FinalizeGStreamerPipeline();

        return false;
    }

    g_object_set(G_OBJECT(pngenc), "snapshot", 0, nullptr);
    gst_bin_add(GST_BIN(pipeline), pngenc);

    // create multifilesink element
    GstElement * multifilesink = gst_element_factory_make("multifilesink", nullptr);
    if (!multifilesink)
    {
        NVXIO_PRINT("Cannot create multifilesink element");
        FinalizeGStreamerPipeline();

        return false;
    }

    g_object_set(G_OBJECT(multifilesink),
                 "location", windowTitle_.c_str(),
                 "max-lateness", G_GINT64_CONSTANT(-1),
                 "async", FALSE,
                 "render-delay", G_GUINT64_CONSTANT(0),
                 "throttle-time", G_GUINT64_CONSTANT(0),
                 "index", 1,
                 "max-files", 9999u,
                 "post-messages", TRUE,
                 "next-file", 0,
                 nullptr);

    gst_bin_add(GST_BIN(pipeline), multifilesink);

#if GST_VERSION_MAJOR == 0
    std::unique_ptr<GstCaps, nvidiaio::GStreamerObjectDeleter> caps_color(
                gst_caps_new_simple("video/x-raw-rgb",
                                    "bpp", G_TYPE_INT, 32,
                                    "depth", G_TYPE_INT, 32,
                                    "endianness", G_TYPE_INT, 4321,
                                    "red_mask", G_TYPE_INT, -16777216,
                                    "green_mask", G_TYPE_INT, 16711680,
                                    "blue_mask", G_TYPE_INT, 65280,
                                    "alpha_mask", G_TYPE_INT, 255,
                                    "width", G_TYPE_INT, wndWidth_,
                                    "height", G_TYPE_INT, wndHeight_,
                                    "framerate", GST_TYPE_FRACTION, GSTREAMER_DEFAULT_FPS, 1,
                                    nullptr));
    if (!caps_color)
    {
        NVXIO_PRINT("Failed to create caps");
        FinalizeGStreamerPipeline();
        return false;
    }

    if (!gst_element_link_filtered(appsrcelem, color, caps_color.get()))
    {
        NVXIO_PRINT("GStreamer: cannot link " COLOR_ELEM
                    " -> videoflip -> pngenc -> multifilesink");
        FinalizeGStreamerPipeline();

        return false;
    }

#else
    if (!gst_element_link(appsrcelem, color))
    {
        NVXIO_PRINT("GStreamer: cannot link appsrc -> " COLOR_ELEM);
        FinalizeGStreamerPipeline();
        return false;
    }
#endif

    if (!gst_element_link_many(color, videoflip,
                               pngenc, multifilesink, nullptr))
    {
        NVXIO_PRINT("GStreamer: cannot link " COLOR_ELEM
                    " -> videoflip -> pngenc -> multifilesink");
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
