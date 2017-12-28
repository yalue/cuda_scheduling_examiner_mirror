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

#ifdef USE_OPENCV

#include <system_error>
#include <map>

#include <cuda_runtime.h>

#include "FrameSource/OpenCV/OpenCVFrameSourceImpl.hpp"

#include <NVX/Application.hpp>
#include <NVX/ProfilerRange.hpp>

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

OpenCVFrameSourceImpl::OpenCVFrameSourceImpl(std::unique_ptr<OpenCVBaseFrameSource> source):
    FrameSource(source->getSourceType(), source->getSourceName()),
    alive_(false),
    source_(std::move(source)),
    queue_(4u),
    deviceID(-1),
    exec_target { },
    devMem(nullptr),
    devMemPitch(0)
{
    CUDA_SAFE_CALL( cudaGetDevice(&deviceID) );
    exec_target.base.exec_target_type = NVXCU_STREAM_EXEC_TARGET;
    exec_target.stream = nullptr;
    CUDA_SAFE_CALL( cudaGetDeviceProperties(&exec_target.dev_prop, deviceID) );
}

bool OpenCVFrameSourceImpl::open()
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "FrameSource::open (NVXIO)");

    if (!source_)
        return false;

    if (alive_)
        close();

    try
    {
        alive_ = source_->open();
    }
    catch (const cv::Exception &)
    {
        alive_ = false;
        source_->close();
    }

    if (alive_)
    {
        try
        {
            thread = std::thread(&OpenCVFrameSourceImpl::threadFunc, this);
            return true;
        }
        catch (std::system_error &)
        {
            alive_ = false;
            source_->close();
        }
    }

    return alive_;
}

FrameSource::Parameters OpenCVFrameSourceImpl::getConfiguration()
{
    return source_->getConfiguration();
}

bool OpenCVFrameSourceImpl::setConfiguration(const FrameSource::Parameters &params)
{
    return source_->setConfiguration(params);
}

FrameSource::FrameStatus OpenCVFrameSourceImpl::fetch(const image_t & image, uint32_t timeout)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "FrameSource::fetch (NVXIO)");

    cv::Mat frame;

    if (queue_.pop(frame, timeout))
    {
        FrameSource::Parameters configuration = source_->getConfiguration();
        NVXIO_ASSERT(static_cast<vx_uint32>(frame.cols) == configuration.frameWidth);
        NVXIO_ASSERT(static_cast<vx_uint32>(frame.rows) == configuration.frameHeight);

        int cn = frame.channels();
        nvxcu_df_image_e format = cn == 1 ? NVXCU_DF_IMAGE_U8 :
            cn == 3 ? NVXCU_DF_IMAGE_RGB : NVXCU_DF_IMAGE_RGBX;

        convertFrame(exec_target,
                     image,
                     configuration,
                     frame.cols, frame.rows,
                     true, frame.step,
                     cn, frame.data,
                     false,
                     devMem,
                     devMemPitch);
        return nvxio::FrameSource::OK;
    }
    else
    {
        if (alive_)
        {
            return nvxio::FrameSource::TIMEOUT;
        }
        else
        {
            close();
            return nvxio::FrameSource::CLOSED;
        }
    }
}

void OpenCVFrameSourceImpl::close()
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "FrameSource::close (NVXIO)");

    alive_ = false;
    if (thread.joinable())
        thread.join();

    queue_.clear();
    source_->close();

    if (devMem)
    {
        cudaFree(devMem);
        devMem = nullptr;
    }
}

OpenCVFrameSourceImpl::~OpenCVFrameSourceImpl()
{
    close();
}

void OpenCVFrameSourceImpl::threadFunc()
{
    const unsigned int timeout = 30; /*milliseconds*/

    while (alive_ && source_->grab())
    {
        cv::Mat tmp = source_->fetch();
        while (alive_ && !queue_.push(tmp, timeout)) { }
    }

    alive_ = false;
}

}

#endif
