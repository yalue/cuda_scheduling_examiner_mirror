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

using nvxio::makeUP;

namespace nvidiaio {

namespace {

//
// Parses the URI in the form: protocol:///path/to/something/?key1=value1&key2=value2&...
//

bool parseURI(const std::string & uri,
              std::string & protocol,
              std::string & path,
              std::map<std::string, std::string> & keyValues)
{
    keyValues.clear();
    std::size_t pos = uri.find(":///");

    // suppose that it's a regular file
    if (pos == std::string::npos)
    {
        NVXIO_PRINT("Treat \"%s\" as a regular file", uri.c_str());

        protocol = "file";
        path = uri;

        return true;
    }

    // extract protocol
    protocol = uri.substr(0, pos);
    NVXIO_PRINT("Found protocol: \"%s\"", protocol.c_str());

    if (protocol != "file" && protocol != "device")
    {
        NVXIO_PRINT("Unknown protocol specified: \"%s\"", protocol.c_str());
        return false;
    }

    // extract path
    std::string tmp = uri.substr(pos + 4);

    if (protocol == "file")
    {
        path = tmp;
        return true;
    }

    pos = tmp.find('?');

    // tmp is a path
    if (pos == std::string::npos)
    {
        NVXIO_PRINT("Found path: \"%s\"", tmp.c_str());

        path = tmp;
        return true;
    }

    path = tmp.substr(0, pos);
    NVXIO_PRINT("Found path: \"%s\"", path.c_str());

    // parse key=value
    tmp = tmp.substr(pos + 1);
    NVXIO_PRINT("Parse an array of key=value: \"%s\"", tmp.c_str());

    while (!tmp.empty())
    {
        std::size_t pos = tmp.find('&');
        std::string keyValue;

        // it's the last pair
        if (pos == std::string::npos)
            keyValue = tmp;
        else
            keyValue = tmp.substr(0, pos);

        if (!keyValue.empty())
        {
            std::size_t equalPos = keyValue.find('=');

            if (equalPos == std::string::npos)
                NVXIO_THROW_EXCEPTION("The \"" << keyValue << "\" key is specified without a value");

            std::string key = keyValue.substr(0, equalPos),
                    value = keyValue.substr(equalPos + 1);

            if (key.empty())
                NVXIO_THROW_EXCEPTION("Empty key specified");

            if (value.empty())
                NVXIO_THROW_EXCEPTION("Empty value specified");

            if (!keyValues[key].empty())
                NVXIO_THROW_EXCEPTION("The \"" << key << "\" is specified multiple times");

            NVXIO_PRINT("Found key \"%s\" with the value \"%s\"", key.c_str(), value.c_str());
            keyValues[key] = value;
        }

        // move to the next pair
        if (pos == std::string::npos)
            break;

        tmp = tmp.substr(pos + 1);
    }

    return true;
}

int resolveIntegerValue(std::map<std::string, std::string> keyValues, const std::string & key)
{
    int idx = -1;
    std::string & value = keyValues[key];

    if (value.empty())
        NVXIO_THROW_EXCEPTION("Mandatory key \"" << key << "\" is not found");

    if (sscanf(value.c_str(), "%d", &idx) != 1)
        NVXIO_THROW_EXCEPTION("Failed to convert \"" << key << "\" value to native representation");

    return idx;
}

} // namespace

std::unique_ptr<FrameSource> createDefaultFrameSource(const std::string& uri)
{
    std::string protocol, path;
    std::map<std::string, std::string> keyValues;

    if (!parseURI(uri, protocol, path, keyValues))
    {
        NVXIO_PRINT("Failed to parse URI");
        return nullptr;
    }

    if (protocol.empty() || protocol == "file")
    {
        if (!path.empty())
        {
            std::string ext = path.substr(path.rfind(".") + 1);
            // cppcheck-suppress duplicateBranch
            if ((ext == std::string("png")) ||
                (ext == std::string("jpg")) ||
                (ext == std::string("jpeg")) ||
                (ext == std::string("bmp")) ||
                (ext == std::string("tiff")))
            {
#if defined USE_GSTREAMER || defined USE_OPENCV
                size_t pos = path.find('%');
                bool isImageSequence = pos != std::string::npos;

#ifdef USE_GSTREAMER
                return makeUP<GStreamerImagesFrameSourceImpl>(isImageSequence ? nvxio::FrameSource::IMAGE_SEQUENCE_SOURCE :
                                                                                nvxio::FrameSource::SINGLE_IMAGE_SOURCE, path);

#endif
#ifdef USE_OPENCV
                std::unique_ptr<OpenCVBaseFrameSource> ocvSource;

                if (isImageSequence)
                    ocvSource.reset(new OpenCVVideoFrameSource(path, true));
                else
                    ocvSource.reset(new OpenCVImageFrameSource(path));

                if (ocvSource)
                    return makeUP<OpenCVFrameSourceImpl>(std::move(ocvSource));
#endif
#endif // defined USE_GSTREAMER || defined USE_OPENCV
            }
            else
            {
#ifdef USE_NVMEDIA
                if (ext == std::string("h264"))
                    return makeUP<NvMediaVideoFrameSourceImpl>(path);
#endif
#ifdef USE_GSTREAMER
# ifdef USE_GSTREAMER_NVMEDIA
                return makeUP<GStreamerNvMediaFrameSourceImpl>(path);
# else
#  if defined USE_GSTREAMER_OMX && defined USE_GLES // For L4T R23 and R24 only
                return makeUP<GStreamerOpenMAXFrameSourceImpl>(path);
#  else
                return makeUP<GStreamerVideoFrameSourceImpl>(path);
#  endif
# endif
#else
#ifdef USE_OPENCV
                std::unique_ptr<OpenCVVideoFrameSource> ocvSource(new OpenCVVideoFrameSource(path, false));
                if (ocvSource)
                    return makeUP<OpenCVFrameSourceImpl>(std::move(ocvSource));
#endif
#endif
            }
        }
    }
    else if (protocol == "device")
    {
        if (path == "nvmedia")
        {
#ifdef USE_NVMEDIA
            int cameraNumber = resolveIntegerValue(keyValues, "number");

            std::string config = keyValues["config"];
            if (config.empty())
                NVXIO_THROW_EXCEPTION("Mandatory key \"config\" is not found");

            if (config.find("dvp-ov10640") != std::string::npos)
            {
#ifdef USE_CSI_OV10640
                return makeUP<NvMediaCSI10640CameraFrameSourceImpl>(config, cameraNumber);
#endif
                NVXIO_PRINT("CSI Omni Vision 10640 camera source is not available on this platform");
                return nullptr;
            }

            if (config.find("dvp-ov10635") != std::string::npos)
            {
#ifdef USE_CSI_OV10635
                return makeUP<NvMediaCSI10635CameraFrameSourceImpl>(config, cameraNumber);
#endif
                NVXIO_PRINT("CSI Omni Vision 10635 camera source is not available on this platform");
                return nullptr;
            }

            (void)cameraNumber;
#endif // USE_NVMEDIA
        }
        else if (path == "nvcamera")
        {
#ifdef USE_NVGSTCAMERA
            return makeUP<GStreamerNvCameraFrameSourceImpl>(0);
#else
            NVXIO_PRINT("NvCamera source is not available on this platform");
            return nullptr;
#endif
        }
        else if (path == "v4l2")
        {
#if defined USE_GSTREAMER || defined USE_OPENCV
            int idx = resolveIntegerValue(keyValues, "index");
#ifdef USE_GSTREAMER
            return makeUP<GStreamerCameraFrameSourceImpl>(static_cast<uint>(idx));
#endif

#ifdef USE_OPENCV
            std::unique_ptr<OpenCVVideoFrameSource> ocvSource(new OpenCVVideoFrameSource(idx));
            if (ocvSource)
                return makeUP<OpenCVFrameSourceImpl>(std::move(ocvSource));
#endif
#endif // defined USE_GSTREAMER || defined USE_OPENCV
        }
    }

    return nullptr;
}

} // namespace nvidiaio


namespace nvxio {

std::unique_ptr<FrameSource> createDefaultFrameSource(const std::string& uri)
{
    std::unique_ptr<nvidiaio::FrameSource> ptr =
            nvidiaio::createDefaultFrameSource(uri);

    if (!ptr)
        return nullptr;

    return makeUP<FrameSourceWrapper>(std::move(ptr));
}

nvxcu_pitch_linear_image_t loadImageFromFile(const std::string& fileName, nvxcu_df_image_e format)
{
    NVXIO_ASSERT((format == NVXCU_DF_IMAGE_NV12) ||
                 (format == NVXCU_DF_IMAGE_RGB) ||
                 (format == NVXCU_DF_IMAGE_RGBX) ||
                 (format == NVXCU_DF_IMAGE_U8));

    std::unique_ptr<FrameSource> frameSource = createDefaultFrameSource(fileName);
    if (!frameSource)
    {
        NVXIO_THROW_EXCEPTION("Cannot create frame source for file: " << fileName);
    }

    if (frameSource->getSourceType() != FrameSource::SINGLE_IMAGE_SOURCE)
    {
        NVXIO_THROW_EXCEPTION("Expected " << fileName << " to be an image");
    }

    FrameSource::Parameters frameConfig = frameSource->getConfiguration();
    frameConfig.format = format;
    frameSource->setConfiguration(frameConfig);

    if (!frameSource->open())
    {
        NVXIO_THROW_EXCEPTION("Cannot open file: " << fileName);
    }

    frameConfig = frameSource->getConfiguration();

    nvxcu_pitch_linear_image_t image = { };

    image.base.width = frameConfig.frameWidth;
    image.base.height = frameConfig.frameHeight;
    image.base.format = format;
    image.base.image_type = NVXCU_PITCH_LINEAR_IMAGE;

    uint32_t planes = format == NVXCU_DF_IMAGE_NV12 ? 2u : 1u;
    uint32_t channels = format == NVXCU_DF_IMAGE_NV12 ? 1u :
                        format == NVXCU_DF_IMAGE_RGB ? 3u :
                        format == NVXCU_DF_IMAGE_RGBX ? 4u : 1u;

    for (uint32_t p = 0u; p < planes; ++p)
    {
        uint32_t width = image.base.width, height = image.base.width;

        if (p == 1u)
        {
            width >>= 1u;
            height >>= 1u;
            channels = 2u;
        }

        width *= channels;

        size_t pitch = 0ul;
        NVXIO_CUDA_SAFE_CALL( cudaMallocPitch(&image.planes[p].dev_ptr, &pitch,
                              width, height) );

        image.planes[p].pitch_in_bytes = static_cast<uint32_t>(pitch);
    }

    if (frameSource->fetch(image, nvxio::TIMEOUT_INFINITE) != FrameSource::OK)
    {
        for (uint32_t p = 0u; p < planes; ++p)
        {
            NVXIO_CUDA_SAFE_CALL( cudaFree(image.planes[p].dev_ptr) );
        }

        NVXIO_THROW_EXCEPTION("Cannot fetch a frame from file: " << fileName);
    }

    return image;
}

} // namespace nvxio
