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

#if defined USE_GUI && defined USE_OPENCV

#include "Render/OpenCV/OpenGLOpenCVRenderImpl.hpp"

#include <NVX/ProfilerRange.hpp>

#include <opencv2/imgproc/imgproc.hpp>

nvidiaio::OpenGLOpenCVRenderImpl::OpenGLOpenCVRenderImpl(TargetType type, const std::string & name) :
    GlfwUIImpl(type, name), cvtType(-1)
{
}

bool nvidiaio::OpenGLOpenCVRenderImpl::open(const std::string& path, uint32_t width, uint32_t height, nvxcu_df_image_e format)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render2D::open (NVXIO)");

    if (!GlfwUIImpl::open(path, width, height, format, false, false))
        return false;

    cv::Size size(width, height);

    // create internal buffers
    displayFrameRGBA.create(height, width, CV_8UC4);
    displayFrameBGR.create(height, width, CV_8UC3);

    if (targetType == nvxio::Render::VIDEO_RENDER)
    {
        writer.open(path, CV_FOURCC('m', 'p', '4', 'v'), OPENCV_DEFAULT_FPS, size);
        cvtType = CV_RGBA2BGR;
    }
    else if (targetType == nvxio::Render::IMAGE_RENDER)
    {
        writer.open(path, 0, OPENCV_DEFAULT_FPS, size);
        cvtType = CV_RGBA2BGRA;
    }
    else
        NVXIO_THROW_EXCEPTION("This render can be used only for rendering of images and video files");

    return writer.isOpened();
}

bool nvidiaio::OpenGLOpenCVRenderImpl::flush()
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render2D::flush (NVXIO)");

    NVXIO_ASSERT(writer.isOpened());
    NVXIO_ASSERT(cvtType >= 0);

    OpenGLContextSafeSetter setter(holder_);

    gl_->PixelStorei(GL_PACK_ALIGNMENT, 1);
    NVXIO_CHECK_GL_ERROR();
    gl_->PixelStorei(GL_PACK_ROW_LENGTH, wndWidth_);
    NVXIO_CHECK_GL_ERROR();

    {
        gl_->ReadPixels(0, 0, wndWidth_, wndHeight_, GL_RGBA, GL_UNSIGNED_BYTE, displayFrameRGBA.data);
        NVXIO_CHECK_GL_ERROR();

        // convert the RGBA image into BGR variant
        cv::cvtColor(displayFrameRGBA, displayFrameBGR, cvtType);
        cv::flip(displayFrameBGR, displayFrameBGR, 0);

        // write the current frame to the OpenCV writer
        writer.write(displayFrameBGR);
    }

    // reset state
    gl_->PixelStorei(GL_PACK_ALIGNMENT, 4);
    NVXIO_CHECK_GL_ERROR();
    gl_->PixelStorei(GL_PACK_ROW_LENGTH, 0);
    NVXIO_CHECK_GL_ERROR();

    glfwSwapBuffers(window_);

    clearGlBuffer();

    return true;
}

#endif // USE_GUI && USE_OPENCV
