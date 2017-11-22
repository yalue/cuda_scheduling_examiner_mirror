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

#ifndef FRAMESOURCE_HPP
#define FRAMESOURCE_HPP

#include <memory>
#include <string>

#include "Private/Types.hpp"

#include <NVX/FrameSource.hpp>
#include <NVX/Utility.hpp>

namespace nvidiaio
{

class FrameSource
{
public:
    typedef nvxio::FrameSource::Parameters Parameters;
    typedef nvxio::FrameSource::SourceType SourceType;
    typedef nvxio::FrameSource::FrameStatus FrameStatus;

    virtual bool open() = 0;
    virtual FrameStatus fetch(const image_t & image, uint32_t timeout = 5u /*milliseconds*/) = 0;
    virtual Parameters getConfiguration() = 0;
    virtual bool setConfiguration(const Parameters& params) = 0;
    virtual void close() = 0;
    virtual ~FrameSource()
    {}

    SourceType getSourceType() const
    {
        return sourceType;
    }

    std::string getSourceName() const
    {
        return sourceName;
    }

protected:
    FrameSource(SourceType type = nvxio::FrameSource::UNKNOWN_SOURCE,
                const std::string & name = "Undefined"):
        sourceType(type),
        sourceName(name)
    {}

    const SourceType  sourceType;
    const std::string sourceName;
};

std::unique_ptr<FrameSource> createDefaultFrameSource(const std::string & uri);

} // namespace nvidiaio

#endif // FRAMESOURCE_HPP
