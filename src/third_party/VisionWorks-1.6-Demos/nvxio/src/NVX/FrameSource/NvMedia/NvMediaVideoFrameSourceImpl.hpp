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

#ifndef NVMEDIA_VIDEOFRAMESOURCEIMPL_HPP
#define NVMEDIA_VIDEOFRAMESOURCEIMPL_HPP

#ifdef USE_NVMEDIA

#include <thread>
#include <mutex>
#include <condition_variable>

#include "FrameSource/FrameSourceImpl.hpp"
#include "FrameSource/EGLAPIAccessors.hpp"

#include <nvmedia.h>
#include <nvmedia_eglstream.h>
#include <cuda_runtime.h>
#include <cudaEGL.h>
#include <video_parser.h>

#define MAX_RENDER_SURFACE 4
#define MAX_DECODE_BUFFERS 17
#define MAX_DISPLAY_BUFFERS 4
#define MAX_FRAMES (MAX_DECODE_BUFFERS + MAX_DISPLAY_BUFFERS)

namespace nvidiaio
{

struct RefCountedFrameBuffer
{
    int nRefs;
    NvMediaVideoSurface *videoSurface;
};

// Forward declaration
class NvMediaVideoFrameSourceImpl;

struct SampleAppContext
{
    // Context
    video_parser_context_s *ctx;
    NVDSequenceInfo nvsi;
    NVDParserParams nvdp;

    // Decoder params
    int decodeWidth;
    int decodeHeight;
    int displayWidth;
    int displayHeight;
    NvMediaVideoDecoder *decoder;

    // Picture buffer params
    int nBuffers;
    int nPicNum;
    RefCountedFrameBuffer RefFrame[MAX_FRAMES];

    // Display params
    NvMediaDevice *device;
    NvMediaVideoMixer *mixer;
    float aspectRatio;

    // Rendering params
    NvMediaVideoSurface *renderSurfaces[MAX_RENDER_SURFACE];
    NvMediaVideoSurface *freeRenderSurfaces[MAX_RENDER_SURFACE];
    NvMediaSurfaceType surfaceType;
    NvMediaEGLStreamProducer *producer;

    // EGL params
    EGLStreamKHR eglStream;
    EGLDisplay   eglDisplay;

    CUeglStreamConnection cudaConsumer;

    bool alive;

    NvMediaVideoFrameSourceImpl * frameSource;

    // for sync purpose
    std::mutex mutex;
    std::condition_variable condVariable;
    bool isStarted;
};

class NvMediaVideoFrameSourceImpl :
        public FrameSource
{
public:
    explicit NvMediaVideoFrameSourceImpl(const std::string & path);
    virtual bool open();
    virtual FrameStatus fetch(const image_t & image, uint32_t timeout = 5u /*milliseconds*/);
    virtual Parameters getConfiguration();
    virtual bool setConfiguration(const Parameters& params);
    virtual void close();
    virtual ~NvMediaVideoFrameSourceImpl();

    // Video mixer
    bool VideoMixerInit(int width, int height, int videoWidth, int videoHeight);
    void VideoMixerDestroy();

    void DisplayFrame(RefCountedFrameBuffer *frame);
    void DisplayFlush();
    Parameters configuration;

protected:

    // Handling of EGL display
    bool InitializeEGLDisplay();

    // Handling of EGL stream
    EGLStreamKHR InitializeEGLStream();
    void FinalizeEglStream();

    // CUDA consumer
    bool InitializeEglCudaConsumer();
    void FinalizeEglCudaConsumer();

    // Decoder
    bool InitializeDecoder();
    void FinalizeDecoder();

    void FetchVideoFile();

    void ReleaseFrame(NvMediaVideoSurface *videoSurface);
    NvMediaVideoSurface * GetRenderSurface();

    SampleAppContext context;
    std::string filePath;

    std::thread fetchThread;
    int32_t deviceID;
    nvxcu_stream_exec_target_t exec_target;

private:
    // temporary CUDA buffer
    void * devMem;
    size_t devMemPitch;
};

} // namespace nvidiaio

#endif // USE_NVMEDIA

#endif // NVMEDIA_VIDEOFRAMESOURCEIMPL_HPP
