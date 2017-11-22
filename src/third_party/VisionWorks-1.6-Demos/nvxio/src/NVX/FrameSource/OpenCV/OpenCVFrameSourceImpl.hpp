/*
# Copyright (c) 2014-2015, NVIDIA CORPORATION. All rights reserved.
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

#ifndef OPENCVFRAMESOURCEIMPL_HPP
#define OPENCVFRAMESOURCEIMPL_HPP

#ifdef USE_OPENCV

#include <memory>
#include <thread>

#include <NVX/ThreadSafeQueue.hpp>

#include "FrameSource/OpenCV/OpenCVBaseFrameSource.hpp"

namespace nvidiaio
{

class OpenCVFrameSourceImpl :
        public FrameSource
{
public:
    explicit OpenCVFrameSourceImpl(std::unique_ptr<OpenCVBaseFrameSource> source);
    virtual bool open();
    virtual FrameStatus fetch(const image_t & image, uint32_t timeout = 5u /*milliseconds*/);
    virtual Parameters getConfiguration();
    virtual bool setConfiguration(const Parameters& params);
    virtual void close();
    virtual ~OpenCVFrameSourceImpl();

protected:
    void threadFunc();

    volatile bool alive_;
    std::unique_ptr<OpenCVBaseFrameSource> source_;
    nvxio::ThreadSafeQueue<cv::Mat> queue_;
    std::thread thread;
    //ovxio::ContextGuard context_;
    int32_t deviceID;
    nvxcu_stream_exec_target_t exec_target;

private:
    // temporary CUDA buffer
    void * devMem;
    size_t devMemPitch;
};

}
#endif // USE_OPENCV
#endif // OPENCVFRAMESOURCEIMPL_HPP
