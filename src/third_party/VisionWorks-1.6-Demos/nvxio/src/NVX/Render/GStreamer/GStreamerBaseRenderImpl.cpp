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

#include "Render/GStreamer/GStreamerBaseRenderImpl.hpp"
#include "Private/GStreamerUtils.hpp"

#include <NVX/ProfilerRange.hpp>

nvidiaio::GStreamerBaseRenderImpl::GStreamerBaseRenderImpl(TargetType type, const std::string & name) :
    GlfwUIImpl(type, name),
    pipeline(nullptr), bus(nullptr),
    appsrc(nullptr), num_frames(0ul)
{
}

bool nvidiaio::GStreamerBaseRenderImpl::open(const std::string& title, uint32_t width, uint32_t height, nvxcu_df_image_e format)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render2D::open (NVXIO)");

    if (!GlfwUIImpl::open(title, width, height, format, false, false))
        return false;

    if (!InitializeGStreamerPipeline())
        return false;

    return true;
}

bool nvidiaio::GStreamerBaseRenderImpl::flush()
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render2D::flush (NVXIO)");

    if (!pipeline)
        return false;

    {
        OpenGLContextSafeSetter setterLast(holder_);

        if (glfwWindowShouldClose(window_))
            return false;

        gl_->PixelStorei(GL_PACK_ALIGNMENT, 1);
        NVXIO_CHECK_GL_ERROR();
        gl_->PixelStorei(GL_PACK_ROW_LENGTH, wndWidth_);
        NVXIO_CHECK_GL_ERROR();

        {
            GstClockTime duration = GST_SECOND / (double)GSTREAMER_DEFAULT_FPS;
            GstClockTime timestamp = num_frames * duration;

#if GST_VERSION_MAJOR == 0
            GstBuffer * buffer = gst_buffer_try_new_and_alloc(wndHeight_ * wndWidth_ * 4);
            if (!buffer)
            {
                NVXIO_PRINT("Cannot create GStreamer buffer");
                FinalizeGStreamerPipeline();
                return false;
            }

            gl_->ReadPixels(0, 0, wndWidth_, wndHeight_, GL_RGBA, GL_UNSIGNED_BYTE, GST_BUFFER_DATA (buffer));
            NVXIO_CHECK_GL_ERROR();

            GST_BUFFER_TIMESTAMP(buffer) = timestamp;
            if (!GST_BUFFER_TIMESTAMP_IS_VALID(buffer))
                NVXIO_PRINT("Failed to setup timestamp");
#else
            GstBuffer * buffer = gst_buffer_new_allocate(nullptr, wndHeight_ * wndWidth_ * 4, nullptr);

            GstMapInfo info;
            gst_buffer_map(buffer, &info, GST_MAP_READ);
            gl_->ReadPixels(0, 0, wndWidth_, wndHeight_, GL_RGBA, GL_UNSIGNED_BYTE, info.data);
            gst_buffer_unmap(buffer, &info);

            GST_BUFFER_PTS(buffer) = timestamp;
            if (!GST_BUFFER_PTS_IS_VALID(buffer))
                NVXIO_PRINT("Failed to setup PTS");

            GST_BUFFER_DTS(buffer) = timestamp;
            if (!GST_BUFFER_DTS_IS_VALID(buffer))
                NVXIO_PRINT("Failed to setup DTS");
#endif
            GST_BUFFER_DURATION(buffer) = duration;
            if (!GST_BUFFER_DURATION_IS_VALID(buffer))
                NVXIO_PRINT("Failed to setup duration");

            GST_BUFFER_OFFSET(buffer) = num_frames++;
            if (!GST_BUFFER_OFFSET_IS_VALID(buffer))
                NVXIO_PRINT("Failed to setup offset");

            if (gst_app_src_push_buffer(appsrc, buffer) != GST_FLOW_OK)
            {
                NVXIO_PRINT("Error pushing buffer to GStreamer pipeline");
                FinalizeGStreamerPipeline();
                return false;
            }
        }

        // reset state
        gl_->PixelStorei(GL_PACK_ALIGNMENT, 4);
        NVXIO_CHECK_GL_ERROR();
        gl_->PixelStorei(GL_PACK_ROW_LENGTH, 0);
        NVXIO_CHECK_GL_ERROR();

        glfwSwapBuffers(window_);
    }

    clearGlBuffer();

    return true;
}

void nvidiaio::GStreamerBaseRenderImpl::FinalizeGStreamerPipeline()
{
    if (pipeline)
    {
        if (num_frames > 0)
        {
            gst_app_src_end_of_stream(appsrc);

            NVXIO_PRINT("Flushing the internal appsrc queue... Please, wait...");

            std::unique_ptr<GstMessage, nvidiaio::GStreamerObjectDeleter> msg(
                        gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
                                                   (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS)));

            if (GST_MESSAGE_TYPE(msg.get()) == GST_MESSAGE_ERROR)
                NVXIO_PRINT("Error during GStreamer video writer finalization");
            else if (GST_MESSAGE_TYPE(msg.get()) == GST_MESSAGE_EOS)
                NVXIO_PRINT("Received EOS. Finished output file writing.");
        }

        gst_object_unref(GST_OBJECT(bus));
        bus = nullptr;

        gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(pipeline));
        pipeline = nullptr;
    }
}

void nvidiaio::GStreamerBaseRenderImpl::close()
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render2D::close (NVXIO)");

    FinalizeGStreamerPipeline();

    num_frames = 0;

    GlfwUIImpl::close();
}

nvidiaio::GStreamerBaseRenderImpl::~GStreamerBaseRenderImpl()
{
    close();
}

#endif // USE_GUI && USE_GSTREAMER
