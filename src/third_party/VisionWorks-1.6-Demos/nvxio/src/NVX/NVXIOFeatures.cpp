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

#include <OVX/UtilityOVX.hpp>

#include "FrameSource/NvMedia/NvMediaCameraConfigParams.hpp"

#include <algorithm>

namespace nvxio
{

std::vector<std::string> getSupportedFeatures()
{
    std::vector<std::string> features;

    // Render stuff

    {
#ifdef USE_GUI
        features.push_back( { "render2d:window:opengl" } );
        features.push_back( { "render2d:video:gstreamer" } );
        features.push_back( { "render2d:image:gstreamer" } );

        features.push_back( { "render3d:window:opengl" } );
#endif
#ifdef USE_OPENCV
        features.push_back( { "render2d:window:opencv" } );
        features.push_back( { "render2d:image:opencv" } );
        features.push_back( { "render2d:video:opencv" } );
#endif
    }

    // FrameSource stuff
    {
#ifdef USE_OPENCV
        features.push_back( { "source:video:opencv" } );
        features.push_back( { "source:image:opencv" } );
#endif
#ifdef USE_GSTREAMER
        features.push_back( { "source:image:gstreamer" } );
        features.push_back( { "source:video:gstreamer" } );
        features.push_back( { "source:camera:v4l2:gstreamer" } );
#endif
#ifdef USE_NVGSTCAMERA
        features.push_back( { "source:camera:nvidia:gstreamer" } );
#endif
#ifdef USE_GSTREAMER_NVMEDIA
        features.push_back( { "source:video:nvmedia:gstreamer" } );
#endif
#ifdef USE_GSTREAMER_OMX
        features.push_back( { "source:video:openmax:gstreamer" } );
#endif
#ifdef USE_NVMEDIA
        features.push_back( { "source:video:nvmedia:pure" } );
#endif

#if defined USE_CSI_OV10635 || defined USE_CSI_OV10640
        for (const auto & pair : cameraConfigCollection)
        {
            const std::string & configName = pair.first;
            std::string featureName;

#ifdef USE_CSI_OV10635
            if (configName.find("ov10635") != std::string::npos)
                featureName = "source:camera:nvmedia:pure:" + configName;
#endif
#ifdef USE_CSI_OV10640
            if (configName.find("ov10640") != std::string::npos)
                featureName = "source:camera:nvmedia:pure:" + configName;
#endif
            if (!featureName.empty())
              features.push_back(featureName);
        }
#endif // USE_CSI_OV10635 || USE_CSI_OV10640
    }

    std::sort(features.begin(), features.end());

    return features;
}

}
