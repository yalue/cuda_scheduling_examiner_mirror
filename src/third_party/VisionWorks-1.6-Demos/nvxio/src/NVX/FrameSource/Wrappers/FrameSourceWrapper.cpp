/*
# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

#include "FrameSourceWrapper.hpp"


#include <cstring>
#include <sstream>
#include <iostream>

namespace nvxio
{
#define THROW_EXCEPTION(msg) \
    do { \
        std::ostringstream ostr_; \
        ostr_ << msg; \
        throw std::runtime_error(ostr_.str()); \
    } while(0)

#define ASSERT(cond) \
    do \
    { \
        bool stat = (cond); \
        if (!stat) \
        { \
            THROW_EXCEPTION(#cond << " failure in file " << __FILE__ << " line " << __LINE__); \
        } \
    } while (0)


FrameSourceWrapper::FrameSourceWrapper(std::unique_ptr<nvidiaio::FrameSource> source) :
    FrameSource(static_cast<FrameSource::SourceType>(source->getSourceType()),
                source->getSourceName()),
    source_(std::move(source)), opened(false)
{
    ASSERT(source_!=nullptr);
}

bool FrameSourceWrapper::open()
{
    return opened = source_->open();
}

FrameSource::FrameStatus FrameSourceWrapper::fetch(const nvxcu_pitch_linear_image_t & image, uint32_t timeout)
{
    if (!opened)
    {
        source_->close();
        return FrameSource::CLOSED;
    }

    nvidiaio::FrameSource::FrameStatus status = source_->fetch(nvidiaio::image_t(image), timeout);

    return static_cast<FrameSource::FrameStatus>(status);
}

FrameSource::Parameters FrameSourceWrapper::getConfiguration()
{
    nvidiaio::FrameSource::Parameters cuda_params = source_->getConfiguration();
    FrameSource::Parameters params = { };

    ASSERT(sizeof(cuda_params) == sizeof(params));
    std::memcpy(&params, &cuda_params, sizeof(cuda_params));

    return params;
}

bool FrameSourceWrapper::setConfiguration(const FrameSource::Parameters& params)
{
    nvidiaio::FrameSource::Parameters cuda_params = { };

    ASSERT(sizeof(cuda_params) == sizeof(params));
    std::memcpy(&cuda_params, &params, sizeof(params));

    return source_->setConfiguration(cuda_params);
}

void FrameSourceWrapper::close()
{
    source_->close();
    opened = false;
}

FrameSourceWrapper::~FrameSourceWrapper()
{
    close();
}

} // namespace nvxio
