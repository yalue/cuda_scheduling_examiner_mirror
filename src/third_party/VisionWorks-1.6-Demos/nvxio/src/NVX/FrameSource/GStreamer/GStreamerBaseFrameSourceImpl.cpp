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

#ifdef USE_GSTREAMER

#include <memory>

#include <VX/vx.h>

#include <NVX/ProfilerRange.hpp>

#include "FrameSource/GStreamer/GStreamerBaseFrameSourceImpl.hpp"

#include <cuda_runtime_api.h>

#include <gst/pbutils/missing-plugins.h>
#include <gst/app/gstappsink.h>

#include <cassert>

namespace nvidiaio
{

void convertFrame(nvxcu_stream_exec_target_t &exec_target,
                  const image_t & image,
                  const FrameSource::Parameters & configuration,
                  int width, int height,
                  bool usePitch, size_t pitch,
                  int depth, void * decodedPtr,
                  bool is_cuda,
                  void *& devMem,
                  size_t & devMemPitch);

vx_image wrapNVXIOImage(vx_context context,
                        const image_t & image);

GStreamerBaseFrameSourceImpl::GStreamerBaseFrameSourceImpl(FrameSource::SourceType type, const std::string & name):
    FrameSource(type, name),
    pipeline(nullptr), bus(nullptr),
    end(true),
    deviceID(-1),
    exec_target { },
    sink(nullptr),
    devMem(nullptr),
    devMemPitch(0ul)
{
    CUDA_SAFE_CALL( cudaGetDevice(&deviceID) );
    exec_target.base.exec_target_type = NVXCU_STREAM_EXEC_TARGET;
    exec_target.stream = nullptr;
    CUDA_SAFE_CALL( cudaGetDeviceProperties(&exec_target.dev_prop, deviceID) );
}

void GStreamerBaseFrameSourceImpl::newGstreamerPad(GstElement * /*elem*/, GstPad *pad, gpointer data)
{
    GstElement * color = (GstElement *) data;

    std::unique_ptr<GstPad, GStreamerObjectDeleter> sinkpad(gst_element_get_static_pad (color, "sink"));
    if (!sinkpad)
    {
        NVXIO_PRINT("Gstreamer: no pad named sink");
        return;
    }

    gst_pad_link(pad, sinkpad.get());
}

bool GStreamerBaseFrameSourceImpl::open()
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "FrameSource::open (NVXIO)");

    if (pipeline)
    {
        close();
    }

    if (!InitializeGstPipeLine())
    {
        NVXIO_PRINT("Cannot initialize Gstreamer pipeline");
        return false;
    }

    NVXIO_ASSERT(!end);

    return true;
}

FrameSource::FrameStatus GStreamerBaseFrameSourceImpl::extractFrameParams(
        const FrameSource::Parameters & configuration,
        GstCaps * bufferCaps,
        gint & width, gint & height, gint & fps, gint & depth)
{
    // fail out if no caps
    assert(gst_caps_get_size(bufferCaps) == 1);
    GstStructure * structure = gst_caps_get_structure(bufferCaps, 0);

    // fail out if width or height are 0
    if (!gst_structure_get_int(structure, "width", &width))
    {
        NVXIO_PRINT("Failed to retrieve width");
        return nvxio::FrameSource::CLOSED;
    }
    if (!gst_structure_get_int(structure, "height", &height))
    {
        NVXIO_PRINT("Failed to retrieve height");
        return nvxio::FrameSource::CLOSED;
    }

    NVXIO_ASSERT(configuration.frameWidth == static_cast<uint32_t>(width));
    NVXIO_ASSERT(configuration.frameHeight == static_cast<uint32_t>(height));

    gint num = 0, denom = 1;
    if (!gst_structure_get_fraction(structure, "framerate", &num, &denom))
    {
        NVXIO_PRINT("Cannot query video fps");
        return nvxio::FrameSource::CLOSED;
    }
    else
        fps = static_cast<float>(num) / denom;

    depth = 0;
    const gchar * name = gst_structure_get_name(structure);
    nvxcu_df_image_e vx_format = NVXCU_DF_IMAGE_NONE;

#if GST_VERSION_MAJOR == 0
    if (!name)
        return nvxio::FrameSource::CLOSED;

    if (strcasecmp(name, "video/x-raw-gray") == 0)
    {
        gint bpp = 0;
        if (!gst_structure_get_int(structure, "bpp", &bpp))
        {
            NVXIO_PRINT("Failed to retrieve BPP");
            return nvxio::FrameSource::CLOSED;
        }

        if (bpp == 8)
        {
            depth = 1;
            vx_format = NVXCU_DF_IMAGE_U8;
        }
    }
    else if (strcasecmp(name, "video/x-raw-rgb") == 0)
    {
        gint bpp = 0;
        if (!gst_structure_get_int(structure, "bpp", &bpp))
        {
            NVXIO_PRINT("Failed to retrieve BPP");
            return nvxio::FrameSource::CLOSED;
        }

        if (bpp == 24)
        {
            depth = 3;
            vx_format = NVXCU_DF_IMAGE_RGB;
        }
        else if (bpp == 32)
        {
            depth = 4;
            vx_format = NVXCU_DF_IMAGE_RGBX;
        }
    }
    else if (strcasecmp(name, "video/x-raw-yuv") == 0)
    {
        guint32 fourcc = 0u;

        if (!gst_structure_get_fourcc(structure, "format", &fourcc))
        {
            NVXIO_PRINT("Failed to retrieve FOURCC");
            return nvxio::FrameSource::CLOSED;
        }

        if (fourcc == GST_MAKE_FOURCC('N', 'V', '1', '2'))
            vx_format = NVXCU_DF_IMAGE_NV12;
    }
#else
    const gchar * format = gst_structure_get_string(structure, "format");

    if (!name || !format)
        return nvxio::FrameSource::CLOSED;

    if (strcasecmp(name, "video/x-raw") == 0)
    {
        if (strcasecmp(format, "RGBA") == 0)
        {
            vx_format = NVXCU_DF_IMAGE_RGBX;
            depth = 4;
        }
        else if (strcasecmp(format, "RGB") == 0)
        {
            vx_format = NVXCU_DF_IMAGE_RGB;
            depth = 3;
        }
        else if (strcasecmp(format, "GRAY8") == 0)
        {
            vx_format = NVXCU_DF_IMAGE_U8;
            depth = 1;
        }
        else if (strcasecmp(format, "NV12") == 0)
            vx_format = NVXCU_DF_IMAGE_NV12;
    }
#endif

    NVXIO_ASSERT(configuration.format == NVXCU_DF_IMAGE_NONE ||
                 configuration.format == vx_format);

    return nvxio::FrameSource::OK;
}

FrameSource::FrameStatus GStreamerBaseFrameSourceImpl::fetch(const image_t & image, uint32_t /*timeout*/)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "FrameSource::fetch (NVXIO)");

    handleGStreamerMessages();

    if (gst_app_sink_is_eos(GST_APP_SINK(sink)))
    {
        close();
        return nvxio::FrameSource::CLOSED;
    }

#if GST_VERSION_MAJOR == 0
    std::unique_ptr<GstBuffer, GStreamerObjectDeleter> bufferHolder(
        gst_app_sink_pull_buffer(GST_APP_SINK(sink)));
    GstBuffer* buffer = bufferHolder.get();
#else
    std::unique_ptr<GstSample, GStreamerObjectDeleter> sample;

    if (sampleFirstFrame)
    {
        sample = std::move(sampleFirstFrame);
        NVXIO_ASSERT(!sampleFirstFrame);
    }
    else
        sample.reset(gst_app_sink_pull_sample(GST_APP_SINK(sink)));

    if (!sample)
    {
        close();
        return nvxio::FrameSource::CLOSED;
    }

    GstBuffer * buffer = gst_sample_get_buffer(sample.get());
#endif

#if GST_VERSION_MAJOR == 0
    std::unique_ptr<GstCaps, GStreamerObjectDeleter> bufferCapsHolder(gst_buffer_get_caps(buffer));
    GstCaps * bufferCaps = bufferCapsHolder.get();
#else
    GstCaps * bufferCaps = gst_sample_get_caps(sample.get());
#endif

    gint width, height, fps, depth;
    if (extractFrameParams(configuration, bufferCaps, width, height,
                           fps, depth) == nvxio::FrameSource::CLOSED)
    {
        close();
        return nvxio::FrameSource::CLOSED;
    }

#if GST_VERSION_MAJOR == 0
    void * decodedPtr = GST_BUFFER_DATA(buffer);
#else
    GstMapInfo info;

    gboolean success = gst_buffer_map(buffer, &info, (GstMapFlags)GST_MAP_READ);
    if (!success)
    {
        NVXIO_PRINT("GStreamer: unable to map buffer");
        close();
        return nvxio::FrameSource::CLOSED;
    }

    void * decodedPtr = info.data;
#endif

    convertFrame(exec_target,
                 image,
                 configuration,
                 width, height,
                 false, 0,
                 depth, decodedPtr,
                 false,
                 devMem,
                 devMemPitch);

#if GST_VERSION_MAJOR != 0
    gst_buffer_unmap(buffer, &info);
#endif

    return nvxio::FrameSource::OK;
}

FrameSource::Parameters GStreamerBaseFrameSourceImpl::getConfiguration()
{
    return configuration;
}

bool GStreamerBaseFrameSourceImpl::setConfiguration(const FrameSource::Parameters& params)
{
    NVXIO_ASSERT(end);

    bool result = true;

    // ignore FPS, width, height values
    if (params.frameWidth != (uint32_t)-1)
        result = false;
    if (params.frameHeight != (uint32_t)-1)
        result = false;
    if (params.fps != (uint32_t)-1)
        result = false;

    NVXIO_ASSERT((params.format == NVXCU_DF_IMAGE_NV12) ||
                 (params.format == NVXCU_DF_IMAGE_U8) ||
                 (params.format == NVXCU_DF_IMAGE_RGB) ||
                 (params.format == NVXCU_DF_IMAGE_RGBX)||
                 (params.format == NVXCU_DF_IMAGE_NONE));

    configuration.format = params.format;

    return result;
}

void GStreamerBaseFrameSourceImpl::close()
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "FrameSource::close (NVXIO)");

    handleGStreamerMessages();
    FinalizeGstPipeLine();

    if (devMem)
    {
        cudaFree(devMem);
        devMem = nullptr;
    }
}

void GStreamerBaseFrameSourceImpl::handleGStreamerMessages()
{
    std::unique_ptr<GstMessage, GStreamerObjectDeleter> msg;
    GError *err = nullptr;
    gchar *debug = nullptr;
    GstStreamStatusType tp;
    GstElement * elem = nullptr;

    if (!bus)
        return;

    while (gst_bus_have_pending(bus))
    {
        msg.reset(gst_bus_pop(bus));

        if (gst_is_missing_plugin_message(msg.get()))
        {
            NVXIO_PRINT("GStreamer: your gstreamer installation is missing a required plugin!");
            end = true;
        }
        else
        {
            switch (GST_MESSAGE_TYPE(msg.get()))
            {
                case GST_MESSAGE_STATE_CHANGED:
                    GstState oldstate, newstate, pendstate;
                    gst_message_parse_state_changed(msg.get(), &oldstate, &newstate, &pendstate);
                    break;
                case GST_MESSAGE_ERROR:
                {
                    gst_message_parse_error(msg.get(), &err, &debug);
                    std::unique_ptr<char[], GlibDeleter> name(gst_element_get_name(GST_MESSAGE_SRC(msg.get())));

                    NVXIO_PRINT("GStreamer Plugin: Embedded video playback halted; module %s reported: %s",
                           name.get(), err->message);

                    g_error_free(err);
                    g_free(debug);
                    end = true;
                    break;
                }
                case GST_MESSAGE_EOS:
                    end = true;
                    break;
                case GST_MESSAGE_STREAM_STATUS:
                    gst_message_parse_stream_status(msg.get(), &tp, &elem);
                    break;
                default:
                    break;
            }
        }
    }
}

void GStreamerBaseFrameSourceImpl::FinalizeGstPipeLine()
{
    if (pipeline)
    {
        handleGStreamerMessages();

        gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_NULL);
        handleGStreamerMessages();

        gst_object_unref(GST_OBJECT(bus));
        bus = nullptr;

        gst_object_unref(GST_OBJECT(pipeline));
        pipeline = nullptr;
    }
}

GStreamerBaseFrameSourceImpl::~GStreamerBaseFrameSourceImpl()
{
    close();
}

} // namespace nvidiaio

#endif // USE_GSTREAMER
