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

#if defined USE_GSTREAMER_OMX && defined USE_GLES

#include "GStreamerOpenMAXFrameSourceImpl.hpp"
#include "GStreamerBaseFrameSourceImpl.hpp"

#include <gst/app/gstappsink.h>
#include <sstream>


namespace nvidiaio
{

GStreamerOpenMAXFrameSourceImpl::GStreamerOpenMAXFrameSourceImpl(const std::string & filename) :
    GStreamerEGLStreamSinkFrameSourceImpl(nvxio::FrameSource::VIDEO_SOURCE, "GStreamerOpenMAXFrameSource", true),
    fileName(filename)
{
}

GstAutoplugSelectResult GStreamerOpenMAXFrameSourceImpl::autoPlugSelect(GstElement *, GstPad *,
                              GstCaps * caps, GstElementFactory *, gpointer)
{
    std::unique_ptr<char[], GlibDeleter> capsStr(gst_caps_to_string(caps));

    if (strstr(capsStr.get(), "video"))
    {
        return GST_AUTOPLUG_SELECT_TRY;
    }
    else
    {
        return GST_AUTOPLUG_SELECT_EXPOSE;
    }
}

bool GStreamerOpenMAXFrameSourceImpl::InitializeGstPipeLine()
{
    GstStateChangeReturn status;
    end = true;

    std::string uri;
    if (!gst_uri_is_valid(fileName.c_str()))
    {
        char* real = realpath(fileName.c_str(), nullptr);

        if (!real)
        {
            NVXIO_PRINT("Can't resolve path \"%s\": %s", fileName.c_str(), strerror(errno));
            return false;
        }

        std::unique_ptr<char[], GlibDeleter> pUri(g_filename_to_uri(real, nullptr, nullptr));
        free(real);
        uri = pUri.get();
    }
    else
    {
        uri = fileName;
    }

    pipeline = GST_PIPELINE(gst_pipeline_new(nullptr));
    if (!pipeline)
    {
        NVXIO_PRINT("Cannot create Gstreamer pipeline");
        return false;
    }

    bus = gst_pipeline_get_bus(GST_PIPELINE (pipeline));

    // create uridecodebin
    GstBin * uriDecodeBin = GST_BIN(gst_element_factory_make("uridecodebin", nullptr));
    if (!uriDecodeBin)
    {
        NVXIO_PRINT("Cannot create uridecodebin");
        FinalizeGstPipeLine();

        return false;
    }

    g_object_set(G_OBJECT(uriDecodeBin),
                 "uri", uri.c_str(),
                 "message-forward", TRUE,
                 nullptr);

    gst_bin_add(GST_BIN(pipeline), GST_ELEMENT(uriDecodeBin));

    // create nvvidconv
    GstElement * nvvidconv = gst_element_factory_make("nvvidconv", nullptr);
    if (!nvvidconv)
    {
        NVXIO_PRINT("Cannot create nvvidconv");
        FinalizeGstPipeLine();

        return false;
    }

    gst_bin_add(GST_BIN(pipeline), nvvidconv);

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
                 nullptr);

    gst_bin_add(GST_BIN(pipeline), nvvideosink);

    g_signal_connect(uriDecodeBin, "autoplug-select", G_CALLBACK(GStreamerOpenMAXFrameSourceImpl::autoPlugSelect), nullptr);
    g_signal_connect(uriDecodeBin, "pad-added", G_CALLBACK(GStreamerBaseFrameSourceImpl::newGstreamerPad), nvvidconv);

    std::unique_ptr<GstCaps, GStreamerObjectDeleter> caps_nvvidconv(
        gst_caps_from_string("video/x-raw(memory:NVMM), format=(string){I420}"));

    // link nvvidconv using caps
    if (!gst_element_link_filtered(nvvidconv, nvvideosink, caps_nvvidconv.get()))
    {
        NVXIO_PRINT("GStreamer: cannot link nvvidconv -> nvvideosink");
        FinalizeGstPipeLine();

        return false;
    }

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

    // GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "gst_pipeline");

    if (!updateConfiguration(nvvideosink, nvvidconv, configuration))
    {
        FinalizeGstPipeLine();
        return false;
    }

    end = false;

    return true;
}

} // namespace nvidiaio

#endif // defined USE_GSTREAMER_OMX && defined USE_GLES
