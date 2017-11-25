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

#ifndef NVMEDIA_CSI10635CAMERAFRAMESOURCEIMPL_HPP
#define NVMEDIA_CSI10635CAMERAFRAMESOURCEIMPL_HPP

#ifdef USE_CSI_OV10635

#include <VX/vx.h>

#include "FrameSource/FrameSourceImpl.hpp"

#include "FrameSource/NvMedia/OV10635/ImageCapture.hpp"

namespace nvidiaio
{

class NvMediaCSI10635CameraFrameSourceImpl :
        public FrameSource
{
public:
    NvMediaCSI10635CameraFrameSourceImpl(const std::string & configName, int number);
    virtual bool open();
    virtual FrameStatus fetch(const image_t & image, uint32_t timeout = 5u /*milliseconds*/);
    virtual Parameters getConfiguration();
    virtual bool setConfiguration(const Parameters& params);
    virtual void close();
    virtual ~NvMediaCSI10635CameraFrameSourceImpl();

protected:
    const char * defaultCameraConfig() const;
    std::string parseCameraConfig(const std::string& cameraConfigFile, CaptureConfigParams& captureConfigCollection);

    // camera params
    CaptureConfigParams captureConfigCollection;
    ov10635::ImgCapture * context;
    int cameraNumber;
    std::string configPath;

    Parameters configuration;
    ovxio::ContextGuard vxContext;
};

} // namespace nvidiaio

#endif // USE_CSI_OV10635

#endif // NVMEDIA_CSI10635CAMERAFRAMESOURCEIMPL_HPP
