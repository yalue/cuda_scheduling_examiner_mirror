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

#ifndef NVMEDIA_CAMERA_CONFIG_PARAMS_HPP
#define NVMEDIA_CAMERA_CONFIG_PARAMS_HPP

#if defined USE_CSI_OV10635 || defined USE_CSI_OV10640

#include <string>
#include <map>
#include <cstring>

#include <nvcommon.h>
#include <nvmedia_image.h>

struct CaptureConfigParams
{
    std::string name;
    std::string description;
    std::string board;
    std::string inputDevice;
    std::string inputFormat;
    std::string surfaceFormat;
    std::string resolution;
    std::string interface;
    int   i2cDevice;
    NvU32 csiLanes;
    NvU32 embeddedDataLinesTop;
    NvU32 embeddedDataLinesBottom;
    NvU32 desAddr;
    NvU32 brdcstSerAddr;
    NvU32 serAddr[NVMEDIA_MAX_AGGREGATE_IMAGES];
    NvU32 brdcstSensorAddr;
    NvU32 sensorAddr[NVMEDIA_MAX_AGGREGATE_IMAGES];

    CaptureConfigParams()
    {
        i2cDevice = 0;

        csiLanes = 0u;
        embeddedDataLinesTop = 0u;
        embeddedDataLinesBottom = 0u;
        desAddr = 0u;
        brdcstSensorAddr = 0u;

        std::memset(sensorAddr, 0, sizeof(sensorAddr));
        std::memset(serAddr, 0, sizeof(serAddr));
    }
};

// contains preset of NvMedia OV 10635 and OV 10640 camera configs
extern std::map<std::string, CaptureConfigParams> cameraConfigCollection;

#endif // USE_CSI_OV10635 || USE_CSI_OV10640

#endif // NVMEDIA_CAMERA_CONFIG_PARAMS_HPP
