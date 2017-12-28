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

#if defined USE_GSTREAMER_OMX && defined USE_GLES || defined USE_GSTREAMER_NVMEDIA || defined USE_NVGSTCAMERA

#include <NVX/ProfilerRange.hpp>

#include "FrameSource/GStreamer/GStreamerEGLStreamSinkFrameSourceImpl.hpp"

#include <cuda_runtime.h>

#include <gst/pbutils/missing-plugins.h>

#include <memory>
#include <thread>
#include <string>

using namespace nvidiaio::egl_api;

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

GStreamerEGLStreamSinkFrameSourceImpl::GStreamerEGLStreamSinkFrameSourceImpl(FrameSource::SourceType sourceType,
                                                                             const char * const name, bool fifomode) :
    FrameSource(sourceType, name),
    pipeline(nullptr),
    bus(nullptr),
    end(true),
    fifoLength(4),
    fifoMode(fifomode),
    latency(0),
    cudaConnection(nullptr),
    deviceID(-1),
    exec_target { },
    nv12Frame(nullptr),
    nv12FramePitch(0ul),
    devMem(nullptr),
    devMemPitch(0ul)
{
    context.stream = EGL_NO_STREAM_KHR;
    context.display = EGL_NO_DISPLAY;
    CUDA_SAFE_CALL( cudaGetDevice(&deviceID) );
    exec_target.base.exec_target_type = NVXCU_STREAM_EXEC_TARGET;
    exec_target.stream = nullptr;
    CUDA_SAFE_CALL( cudaGetDeviceProperties(&exec_target.dev_prop, deviceID) );
}

bool GStreamerEGLStreamSinkFrameSourceImpl::open()
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "FrameSource::open (NVXIO)");

    if (pipeline)
    {
        close();
    }

    NVXIO_PRINT("Initializing EGL display");
    if (!InitializeEGLDisplay())
    {
        NVXIO_PRINT("Cannot initialize EGL display");
        return false;
    }

    NVXIO_PRINT("Initializing EGL stream");
    if (!InitializeEGLStream())
    {
        NVXIO_PRINT("Cannot initialize EGL Stream");
        return false;
    }

    NVXIO_PRINT("Initializing EGL consumer");
    if (!InitializeEglCudaConsumer())
    {
        NVXIO_PRINT("Cannot initialize CUDA consumer");
        return false;
    }

    NVXIO_PRINT("Creating GStreamer pipeline");
    if (!InitializeGstPipeLine())
    {
        NVXIO_PRINT("Cannot initialize Gstreamer pipeline");
        return false;
    }

    return true;
}

bool GStreamerEGLStreamSinkFrameSourceImpl::InitializeEGLDisplay()
{
    // Obtain the EGL display
    context.display = nvidiaio::EGLDisplayAccessor::getInstance();
    if (context.display == EGL_NO_DISPLAY)
    {
        NVXIO_PRINT("EGL failed to obtain display.");
        return false;
    }

    return true;
}

bool GStreamerEGLStreamSinkFrameSourceImpl::InitializeEglCudaConsumer()
{
    if (cudaSuccess != cudaFree(nullptr))
    {
        NVXIO_PRINT("Failed to initialize CUDA context");
        return false;
    }

    NVXIO_PRINT("Connect CUDA consumer");
    CUresult curesult = cuEGLStreamConsumerConnect(&cudaConnection, context.stream);
    if (CUDA_SUCCESS != curesult)
    {
        NVXIO_PRINT("Connect CUDA consumer ERROR %d", curesult);
        return false;
    }

    return true;
}

bool GStreamerEGLStreamSinkFrameSourceImpl::InitializeEGLStream()
{
    const EGLint streamAttrMailboxMode[] = { EGL_NONE };
    const EGLint streamAttrFIFOMode[] = { EGL_STREAM_FIFO_LENGTH_KHR, fifoLength, EGL_NONE };

    if(!setupEGLExtensions())
        return false;

    context.stream = eglCreateStreamKHR(context.display, fifoMode ? streamAttrFIFOMode : streamAttrMailboxMode);
    if (context.stream == EGL_NO_STREAM_KHR)
    {
        NVXIO_PRINT("Couldn't create stream.");
        return false;
    }

    if (!eglStreamAttribKHR(context.display, context.stream, EGL_CONSUMER_LATENCY_USEC_KHR, latency))
    {
        NVXIO_PRINT("Consumer: streamAttribKHR EGL_CONSUMER_LATENCY_USEC_KHR failed");
    }
    if (!eglStreamAttribKHR(context.display, context.stream, EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR, 0))
    {
        NVXIO_PRINT("Consumer: streamAttribKHR EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR failed");
    }

    // Get stream attributes
    if (!eglQueryStreamKHR(context.display, context.stream, EGL_STREAM_FIFO_LENGTH_KHR, &fifoLength))
    {
        NVXIO_PRINT("Consumer: eglQueryStreamKHR EGL_STREAM_FIFO_LENGTH_KHR failed");
    }
    if (!eglQueryStreamKHR(context.display, context.stream, EGL_CONSUMER_LATENCY_USEC_KHR, &latency))
    {
        NVXIO_PRINT("Consumer: eglQueryStreamKHR EGL_CONSUMER_LATENCY_USEC_KHR failed");
    }

    if (fifoMode != (fifoLength > 0))
    {
        NVXIO_PRINT("EGL Stream consumer - Unable to set FIFO mode");
        fifoMode = false;
    }
    if (fifoMode)
    {
        NVXIO_PRINT("EGL Stream consumer - Mode: FIFO Length: %d", fifoLength);
    }
    else
    {
        NVXIO_PRINT("EGL Stream consumer - Mode: Mailbox");
    }

    return true;
}

FrameSource::FrameStatus GStreamerEGLStreamSinkFrameSourceImpl::fetch(const image_t & image, uint32_t timeout)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "FrameSource::fetch (NVXIO)");

    handleGStreamerMessages();

    if (cudaSuccess != cudaFree(nullptr))
    {
        NVXIO_PRINT("Failed to initialize CUDA context");
        return nvxio::FrameSource::CLOSED;
    }

    CUgraphicsResource cudaResource;
    CUeglFrame eglFrame;
    EGLint streamState = 0;

    if (!eglQueryStreamKHR(context.display, context.stream, EGL_STREAM_STATE_KHR, &streamState))
    {
        NVXIO_PRINT("Cuda consumer, eglQueryStreamKHR EGL_STREAM_STATE_KHR failed");
        close();
        return nvxio::FrameSource::CLOSED;
    }

    if (streamState == EGL_STREAM_STATE_DISCONNECTED_KHR || end)
    {
        NVXIO_PRINT("CUDA Consumer: - EGL_STREAM_STATE_DISCONNECTED_KHR received");
        close();
        return nvxio::FrameSource::CLOSED;
    }

    if (streamState != EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR)
    {
        return nvxio::FrameSource::TIMEOUT;
    }

    CUresult cuStatus = cuEGLStreamConsumerAcquireFrame(&cudaConnection, &cudaResource, nullptr, timeout*1000);
    if (cuStatus != CUDA_SUCCESS)
    {
        NVXIO_PRINT("Cuda Acquire failed cuStatus=%d", cuStatus);
        close();
        return nvxio::FrameSource::CLOSED;
    }

    cuStatus = cuGraphicsResourceGetMappedEglFrame(&eglFrame, cudaResource, 0, 0);
    if (cuStatus != CUDA_SUCCESS)
    {
        NVXIO_PRINT("Cuda get resource failed with %d", cuStatus);
        cuEGLStreamConsumerReleaseFrame(&cudaConnection, cudaResource, nullptr);
        close();
        return nvxio::FrameSource::CLOSED;
    }

    NVXIO_ASSERT(eglFrame.width == configuration.frameWidth);
    NVXIO_ASSERT(eglFrame.height == configuration.frameHeight);

    NVXIO_ASSERT(eglFrame.cuFormat == CU_AD_FORMAT_UNSIGNED_INT8);

    if (eglFrame.eglColorFormat == CU_EGL_COLOR_FORMAT_RGBA)
    {
        NVXIO_ASSERT(eglFrame.planeCount == 1);
        NVXIO_ASSERT(eglFrame.numChannels == 4);
        NVXIO_ASSERT(eglFrame.frameType == CU_EGL_FRAME_TYPE_PITCH);

        configuration.format = NVXCU_DF_IMAGE_RGBX;

        convertFrame(exec_target,
                     image,
                     configuration,
                     eglFrame.width, eglFrame.height,
                     true, eglFrame.pitch,
                     eglFrame.numChannels, eglFrame.frame.pPitch[0],
                     true,
                     devMem,
                     devMemPitch);
    }
    else if (eglFrame.eglColorFormat == CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR)
    {
        NVXIO_ASSERT(eglFrame.planeCount == 2);
        NVXIO_ASSERT(eglFrame.frameType == CU_EGL_FRAME_TYPE_ARRAY);

        if (image.format == NVXCU_DF_IMAGE_NV12)
        {
            vx_int32 stride_y = ((eglFrame.width + 3) >> 2) << 2;

            cudaStream_t stream = nullptr;
            NVXIO_ASSERT( cudaMemcpy2DFromArrayAsync(image.planes[0].ptr, image.planes[0].pitch_in_bytes,
                          (const struct cudaArray *)eglFrame.frame.pArray[0],
                          0, 0,
                          eglFrame.width * sizeof(vx_uint8), eglFrame.height,
                          cudaMemcpyDeviceToDevice, stream) == cudaSuccess );

            // copy the second plane u/v

            NVXIO_ASSERT( (cudaMemcpy2DFromArrayAsync(image.planes[1].ptr, image.planes[1].pitch_in_bytes,
                          (const struct cudaArray *)eglFrame.frame.pArray[1],
                          0, 0,
                          (eglFrame.width >> 1) * sizeof(vx_uint16), eglFrame.height >> 1,
                          cudaMemcpyDeviceToDevice, stream) == cudaSuccess) );

            NVXIO_ASSERT( cudaStreamSynchronize(stream) == cudaSuccess );
        }
        else
        {
            if (!nv12Frame)
            {
                size_t height_dec = eglFrame.height;
                height_dec += height_dec >> 1;

                NVXIO_ASSERT( cudaSuccess == cudaMallocPitch(&nv12Frame, &nv12FramePitch,
                                                             eglFrame.width, height_dec) );
            }
            cudaStream_t stream = nullptr;
            NVXIO_ASSERT( cudaMemcpy2DFromArrayAsync(nv12Frame, nv12FramePitch,
                          (const struct cudaArray *)eglFrame.frame.pArray[0],
                          0, 0,
                          eglFrame.width * sizeof(vx_uint8), eglFrame.height,
                          cudaMemcpyDeviceToDevice, stream) == cudaSuccess );

            // copy the second plane u/v

            NVXIO_ASSERT( (cudaMemcpy2DFromArrayAsync(((uint8_t *)nv12Frame + nv12FramePitch * eglFrame.height), nv12FramePitch,
                          (const struct cudaArray *)eglFrame.frame.pArray[1],
                          0, 0,
                          (eglFrame.width >> 1) * sizeof(vx_uint16), eglFrame.height >> 1,
                          cudaMemcpyDeviceToDevice, stream) == cudaSuccess) );

            NVXIO_ASSERT( cudaStreamSynchronize(stream) == cudaSuccess );

            configuration.format = NVXCU_DF_IMAGE_NV12;

            convertFrame(exec_target,
                         image,
                         configuration,
                         eglFrame.width, eglFrame.height,
                         true, nv12FramePitch,
                         eglFrame.numChannels, nv12Frame,
                         true,
                         devMem,
                         devMemPitch);

        }

    }
    else
    {
        NVXIO_THROW_EXCEPTION("Unsupported decoded image format");
    }

    cuStatus = cuEGLStreamConsumerReleaseFrame(&cudaConnection, cudaResource, nullptr);

    return nvxio::FrameSource::OK;
}

FrameSource::Parameters GStreamerEGLStreamSinkFrameSourceImpl::getConfiguration()
{
    return configuration;
}

bool GStreamerEGLStreamSinkFrameSourceImpl::setConfiguration(const FrameSource::Parameters& params)
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

void GStreamerEGLStreamSinkFrameSourceImpl::handleGStreamerMessages()
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

        if (!msg)
            continue;

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
                    gst_message_parse_state_changed(msg.get(), nullptr, nullptr, nullptr);
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

void GStreamerEGLStreamSinkFrameSourceImpl::FinalizeEglStream()
{
    if (context.stream != EGL_NO_STREAM_KHR)
    {
        eglDestroyStreamKHR(context.display, context.stream);
        context.stream = EGL_NO_STREAM_KHR;
    }
}

void GStreamerEGLStreamSinkFrameSourceImpl::FinalizeEglCudaConsumer()
{
    if (cudaConnection)
    {
        if (cudaSuccess != cudaFree(nullptr))
        {
            NVXIO_PRINT("Failed to initialize CUDA context");
            return;
        }

        cuEGLStreamConsumerDisconnect(&cudaConnection);
        cudaConnection = nullptr;
    }
}

void GStreamerEGLStreamSinkFrameSourceImpl::CloseGstPipeLineAsyncThread()
{
    gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_NULL);
    end = true;
}

void GStreamerEGLStreamSinkFrameSourceImpl::FinalizeGstPipeLine()
{
    if (pipeline)
    {
        std::thread t(&GStreamerEGLStreamSinkFrameSourceImpl::CloseGstPipeLineAsyncThread, this);

        if (fifoMode)
        {
            if (cudaSuccess != cudaFree(nullptr))
            {
                NVXIO_PRINT("Failed to initialize CUDA context");
                return;
            }

            CUgraphicsResource cudaResource;
            EGLint streamState = 0;
            while (!end)
            {
                if (!eglQueryStreamKHR(context.display, context.stream, EGL_STREAM_STATE_KHR, &streamState))
                {
                    handleGStreamerMessages();
                    break;
                }

                if (streamState == EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR)
                {
                    cuEGLStreamConsumerAcquireFrame(&cudaConnection, &cudaResource, nullptr, 1000);
                    cuEGLStreamConsumerReleaseFrame(&cudaConnection, cudaResource, nullptr);
                }
                else
                {
                    handleGStreamerMessages();
                    continue;
                }
                handleGStreamerMessages();
            }

        }

        t.join();

        gst_object_unref(GST_OBJECT(bus));
        bus = nullptr;

        gst_object_unref(GST_OBJECT(pipeline));
        pipeline = nullptr;
    }
}

void GStreamerEGLStreamSinkFrameSourceImpl::close()
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "FrameSource::close (NVXIO)");

    handleGStreamerMessages();
    FinalizeGstPipeLine();
    FinalizeEglCudaConsumer();
    FinalizeEglStream();

    if (nv12Frame)
    {
        cudaFree(nv12Frame);
        nv12Frame = nullptr;
    }

}

GStreamerEGLStreamSinkFrameSourceImpl::~GStreamerEGLStreamSinkFrameSourceImpl()
{
    close();
}

} // namespace nvidiaio

#endif // defined USE_GSTREAMER_OMX && defined USE_GLES || defined USE_GSTREAMER_NVMEDIA || defined USE_NVGSTCAMERA
