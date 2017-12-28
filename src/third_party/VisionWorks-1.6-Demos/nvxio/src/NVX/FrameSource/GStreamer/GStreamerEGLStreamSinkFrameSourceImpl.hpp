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

#ifndef GSTREAMEREGLSTREAMSINKFRAMESOURCEIMPL_HPP
#define GSTREAMEREGLSTREAMSINKFRAMESOURCEIMPL_HPP

#if defined USE_GSTREAMER_OMX && defined USE_GLES || defined USE_GSTREAMER_NVMEDIA || defined USE_NVGSTCAMERA

#include <VX/vx.h>

#include "FrameSource/FrameSourceImpl.hpp"

#include "FrameSource/EGLAPIAccessors.hpp"
#include "FrameSource/GStreamer/GStreamerCommon.hpp"

#include <cudaEGL.h>

namespace nvidiaio
{

class GStreamerEGLStreamSinkFrameSourceImpl :
    public FrameSource
{
public:
    GStreamerEGLStreamSinkFrameSourceImpl(SourceType sourceType, const char * const name, bool fifomode);

    virtual bool open();
    virtual void close();
    virtual FrameStatus fetch(const image_t & image, uint32_t timeout = 5u /*milliseconds*/);

    virtual Parameters getConfiguration();
    virtual bool setConfiguration(const Parameters& params);
    virtual ~GStreamerEGLStreamSinkFrameSourceImpl();

protected:
    void handleGStreamerMessages();

    virtual bool InitializeGstPipeLine() = 0;
    void CloseGstPipeLineAsyncThread();
    void FinalizeGstPipeLine();

    GstPipeline * pipeline;
    GstBus *      bus;
    volatile bool end;

    // EGL context and stream
    struct EglContext
    {
        EGLDisplay display;
        EGLStreamKHR stream;
    };

    bool InitializeEGLDisplay();
    bool InitializeEGLStream();
    void FinalizeEglStream();

    EglContext   context;
    int          fifoLength;
    bool         fifoMode;
    int          latency;

    // CUDA consumer
    bool InitializeEglCudaConsumer();
    void FinalizeEglCudaConsumer();

    CUeglStreamConnection cudaConnection;

    // Common FrameSource parameters
    int32_t deviceID;
    nvxcu_stream_exec_target_t exec_target;

    Parameters configuration;

    // auxilary image
    void * nv12Frame;
    size_t nv12FramePitch;
private:

    // temporary CUDA buffer
    void * devMem;
    size_t devMemPitch;};

} // namespace nvidiaio

#endif // defined USE_GSTREAMER_OMX && defined USE_GLES || defined USE_GSTREAMER_NVMEDIA || defined USE_NVGSTCAMERA

#endif // GSTREAMEREGLSTREAMSINKFRAMESOURCEIMPL_HPP
