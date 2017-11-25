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

#ifdef USE_OPENCV

#include "FrameSource/OpenCV/OpenCVVideoFrameSource.hpp"

#include <opencv2/imgproc/imgproc.hpp>

namespace nvidiaio
{

OpenCVVideoFrameSource::OpenCVVideoFrameSource(int _cameraId):
    OpenCVBaseFrameSource(nvxio::FrameSource::CAMERA_SOURCE, "OpenCVVideoFrameSource"),
    fileName(),
    cameraId(_cameraId)
{
}

OpenCVVideoFrameSource::OpenCVVideoFrameSource(const std::string& _fileName, bool sequence):
    OpenCVBaseFrameSource(sequence ? nvxio::FrameSource::IMAGE_SEQUENCE_SOURCE :
                                     nvxio::FrameSource::VIDEO_SOURCE,
                 "OpenCVVideoFrameSource"),
    fileName(_fileName),
    cameraId(-1)
{
}

bool OpenCVVideoFrameSource::open()
{
    bool opened = false;

    if (fileName.empty())
        opened = capture.open(cameraId);
    else
        opened = capture.open(fileName);

    if (opened)
        updateConfiguration();

    return opened;
}

bool OpenCVVideoFrameSource::setConfiguration(const FrameSource::Parameters& params)
{
    NVXIO_ASSERT(!capture.isOpened());

    bool result = true;

    // ignore FPS, width, height values
    if (params.frameWidth != (uint32_t)-1)
        result = false;
    if (params.frameHeight != (uint32_t)-1)
        result = false;
    if (params.fps != (uint32_t)-1)
        result = false;

    NVXIO_ASSERT((params.format == NVXCU_DF_IMAGE_NV12) ||
                 (params.format == NVXCU_DF_IMAGE_U8) ||
                 (params.format == NVXCU_DF_IMAGE_RGB) ||
                 (params.format == NVXCU_DF_IMAGE_RGBX)||
                 (params.format == NVXCU_DF_IMAGE_NONE));

    configuration.format = params.format;

    return result;
}

void OpenCVVideoFrameSource::updateConfiguration()
{
    configuration.fps = static_cast<uint32_t>(capture.get(CV_CAP_PROP_FPS));
    configuration.frameWidth = static_cast<uint32_t>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
    configuration.frameHeight = static_cast<uint32_t>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));

    int type = capture.get(CV_CAP_PROP_FORMAT), depth = CV_MAT_DEPTH(type);
    NVXIO_ASSERT(depth == CV_8U);

    // the code below gives cn == 1 even for RGB/RGBX frames
    // looks like it's OpenCV issue.

    /*
    int cn = CV_MAT_CN(type);
    configuration.format = cn == 1 ? NVXCU_DF_IMAGE_U8 :
        cn == 3 ? NVXCU_DF_IMAGE_RGB : cn == 4 ? NVXCU_DF_IMAGE_RGBX : NVXCU_DF_IMAGE_NONE;
    */

    // so, let's use RGBX as a default format
    configuration.format = NVXCU_DF_IMAGE_RGBX;
}

FrameSource::Parameters OpenCVVideoFrameSource::getConfiguration()
{
    return configuration;
}

cv::Mat OpenCVVideoFrameSource::fetch()
{
    cv::Mat imageconv;

    if (!capture.retrieve(image))
    {
        close();
        return imageconv;
    }

    // swap channels
    int cn = image.channels();
    if (cn == 3)
        cv::cvtColor(image, imageconv, CV_BGR2RGB);
    else if (cn == 4)
        cv::cvtColor(image, imageconv, CV_BGRA2RGBA);
    else
        image.copyTo(imageconv);

    return imageconv;
}

bool OpenCVVideoFrameSource::grab()
{
    return capture.grab();
}

void OpenCVVideoFrameSource::close()
{
    capture.release();
}

OpenCVVideoFrameSource::~OpenCVVideoFrameSource()
{
    close();
}

}

#endif
