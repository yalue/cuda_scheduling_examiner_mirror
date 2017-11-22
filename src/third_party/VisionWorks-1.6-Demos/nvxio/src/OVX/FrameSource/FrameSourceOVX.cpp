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

#include "Wrappers/FrameSourceOVXWrapper.hpp"
#include "FrameSource/Wrappers/FrameSourceWrapper.hpp"

#ifdef USE_OPENCV
# include "FrameSource/OpenCV/OpenCVFrameSourceImpl.hpp"
# include "FrameSource/OpenCV/OpenCVImageFrameSource.hpp"
# include "FrameSource/OpenCV/OpenCVVideoFrameSource.hpp"
#endif

#ifdef USE_GSTREAMER
# ifdef USE_GSTREAMER_NVMEDIA
#  include "FrameSource/GStreamer/GStreamerNvMediaFrameSourceImpl.hpp"
# endif
# include "FrameSource/GStreamer/GStreamerVideoFrameSourceImpl.hpp"
# include "FrameSource/GStreamer/GStreamerCameraFrameSourceImpl.hpp"
# include "FrameSource/GStreamer/GStreamerImagesFrameSourceImpl.hpp"
# ifdef USE_NVGSTCAMERA
#  include "FrameSource/GStreamer/GStreamerNvCameraFrameSourceImpl.hpp"
# endif
# if defined USE_GSTREAMER_OMX && defined USE_GLES // For L4T R23 and R24 only
#  include "FrameSource/GStreamer/GStreamerOpenMAXFrameSourceImpl.hpp"
# endif
#endif

#ifdef USE_NVMEDIA
# include "FrameSource/NvMedia/NvMediaVideoFrameSourceImpl.hpp"
# ifdef USE_CSI_OV10635
# include "FrameSource/NvMedia/NvMediaCSI10635CameraFrameSourceImpl.hpp"
# endif
# ifdef USE_CSI_OV10640
#  include "FrameSource/NvMedia/NvMediaCSI10640CameraFrameSourceImpl.hpp"
# endif
#endif


#include <NVX/ThreadSafeQueue.hpp>

#include <map>
#include <string>

#include <cuda_runtime_api.h>

#include "Private/LogUtils.hpp"

using ovxio::makeUP;

namespace ovxio {
#include <OVX/UtilityOVX.hpp>
std::unique_ptr<FrameSource> createDefaultFrameSource(vx_context context, const std::string& uri)
{
    checkIfContextIsValid(context);

    std::unique_ptr<nvidiaio::FrameSource> ptr =
            nvidiaio::createDefaultFrameSource(uri);

    if (!ptr)
        return nullptr;

    return makeUP<FrameSourceWrapper>(context, std::move(ptr));
}

vx_image loadImageFromFile(vx_context context, const std::string& fileName, vx_df_image format)
{
    checkIfContextIsValid(context);

    auto frameSource = createDefaultFrameSource(context, fileName);
    if (!frameSource)
    {
        NVXIO_THROW_EXCEPTION("Cannot create frame source for file: " << fileName);
    }

    if (frameSource->getSourceType() != FrameSource::SINGLE_IMAGE_SOURCE)
    {
        NVXIO_THROW_EXCEPTION("Expected " << fileName << " to be an image");
    }

    auto frameConfig = frameSource->getConfiguration();
    frameConfig.format = format;
    frameSource->setConfiguration(frameConfig);

    if (!frameSource->open())
    {
        NVXIO_THROW_EXCEPTION("Cannot open file: " << fileName);
    }

    frameConfig = frameSource->getConfiguration();

    vx_image image = vxCreateImage(context, frameConfig.frameWidth, frameConfig.frameHeight, format);
    NVXIO_CHECK_REFERENCE(image);

    if (frameSource->fetch(image, nvxio::TIMEOUT_INFINITE) != FrameSource::OK)
    {
        NVXIO_SAFE_CALL( vxReleaseImage(&image) );
        NVXIO_THROW_EXCEPTION("Cannot fetch a frame from file: " << fileName);
    }

    return image;
}

} // namespace ovxio
