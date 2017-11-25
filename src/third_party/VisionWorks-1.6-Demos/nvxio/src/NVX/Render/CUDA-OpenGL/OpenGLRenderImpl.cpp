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

#ifdef USE_GUI

#include "OpenGLRenderImpl.hpp"

#include <cmath>
#include <limits>
#include <algorithm>

#ifdef _WIN32
// cuda_gl_interop.h includes GL/gl.h, which requires Windows.h to work.
#define NOMINMAX
#include <windows.h>
#endif

#include <cuda_gl_interop.h>

#include <NVX/ProfilerRange.hpp>
#ifndef __ANDROID__
#include <NVX/Application.hpp>
#endif

#include "RenderUtils.hpp"

const int fontSize = 20;

nvidiaio::OpenGLRenderImpl::OpenGLRenderImpl(TargetType type, const std::string& name) :
    Render(type, name),
    wndWidth_(0u), wndHeight_(0u),
    fboTex_(0u), doScale_(true), textureWidth_(0u), textureHeight_(0u),
    scaleRatioImage_(1.0f), fbo_(0u), fboOld_(0u), imagePut_(false)
{
}

uint32_t nvidiaio::OpenGLRenderImpl::getViewportWidth() const
{
    if (!imagePut_)
        NVXIO_THROW_EXCEPTION("You have to `putImage` first before invoking this method");

    return textureWidth_;
}

uint32_t nvidiaio::OpenGLRenderImpl::getViewportHeight() const
{
    if (!imagePut_)
        NVXIO_THROW_EXCEPTION("You have to `putImage` first before invoking this method");

    return textureHeight_;
}

void nvidiaio::OpenGLRenderImpl::putImage(const image_t & image)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render2D::putImage (NVXIO)");

    // set flag; now use can get proper window width and height
    imagePut_ = true;

    uint32_t imageWidth = image.width, imageHeight = image.height;

    // calculate actual ScaleRatio that will be applied to other primitives like lines, circles, etc.

    bool imageIsBiggerThanWindow = (wndWidth_ < imageWidth || wndHeight_ < imageHeight);
    if (imageIsBiggerThanWindow)
    {
        NVXIO_PRINT("Image size (%u x %u) is bigger then window (%u x %u). Do scaling to fit in window",
                    imageWidth, imageHeight, wndWidth_, wndHeight_);
    }

    // scale
    if (doScale_ || imageIsBiggerThanWindow)
    {
        float widthRatio = static_cast<float>(wndWidth_) / imageWidth;
        float heightRatio = static_cast<float>(wndHeight_) / imageHeight;
        scaleRatioImage_ = std::min(widthRatio, heightRatio);

        textureWidth_ = static_cast<uint32_t>(scaleRatioImage_ * imageWidth);
        textureHeight_ = static_cast<uint32_t>(scaleRatioImage_ * imageHeight);
    }
    else
    {
        scaleRatioImage_ = 1.0f;
        textureWidth_ = imageWidth;
        textureHeight_ = imageHeight;
    }

    {
        OpenGLContextSafeSetter setter(holder_);

        if (image.format == NVXCU_DF_IMAGE_NV12)
            nv12imageRender_.render(image, textureWidth_, textureHeight_);
        else
            imageRender_.render(image, textureWidth_, textureHeight_);
    }
}

void nvidiaio::OpenGLRenderImpl::putFeatures(const array_t & location, const FeatureStyle& style)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render2D::putFeatures (NVXIO)");

    OpenGLContextSafeSetter setter(holder_);
    featuresRender_.render(location, style, textureWidth_, textureHeight_, scaleRatioImage_);
}

void nvidiaio::OpenGLRenderImpl::putFeatures(const array_t & location, const array_t & styles)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render2D::putFeatures (NVXIO)");

    OpenGLContextSafeSetter setter(holder_);
    featuresRender_.render(location, styles, textureWidth_, textureHeight_, scaleRatioImage_);
}

void nvidiaio::OpenGLRenderImpl::putLines(const array_t & lines, const LineStyle& style)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render2D::putLines (NVXIO)");

    OpenGLContextSafeSetter setter(holder_);
    linesRender_.render(lines, style, textureWidth_, textureHeight_, scaleRatioImage_);
}

void nvidiaio::OpenGLRenderImpl::putMotionField(const image_t & field, const MotionFieldStyle& style)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render2D::putMotionField (NVXIO)");

    OpenGLContextSafeSetter setter(holder_);
    motionFieldRender_.render(field, style, textureWidth_, textureHeight_, scaleRatioImage_);
}

void nvidiaio::OpenGLRenderImpl::putCircles(const array_t & circles, const CircleStyle& style)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render2D::putCircles (NVXIO)");

    OpenGLContextSafeSetter setter(holder_);
    circlesRender_.render(circles, style, textureWidth_, textureHeight_, scaleRatioImage_);
}

void nvidiaio::OpenGLRenderImpl::putTextViewport(const std::string& text, const TextBoxStyle& style)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render2D::putTextViewport (NVXIO)");

    OpenGLContextSafeSetter setter(holder_);
    textRender_.render(text, style, textureWidth_, textureHeight_, 1.0f);
}

void nvidiaio::OpenGLRenderImpl::putObjectLocation(const nvxcu_rectangle_t& location, const DetectedObjectStyle& style)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render2D::putObjectLocation (NVXIO)");

    if (style.radius > 0 && style.isHalfTransparent)
    {
        NVXIO_THROW_EXCEPTION("'Rounded corners' and 'half-transparent' modes are mutually exclusive now.");
    }

    // test location correctness
    NVXIO_ASSERT(location.start_x <= location.end_x);
    NVXIO_ASSERT(location.start_y <= location.end_y);

    OpenGLContextSafeSetter setter(holder_);

    // get minumum and maximum supported line widths
    GLfloat widths[2];
    gl_->GetFloatv(GL_ALIASED_LINE_WIDTH_RANGE, widths);
    NVXIO_CHECK_GL_ERROR();

    GLuint lineWidth = std::max<GLuint>(std::min<GLuint>(widths[1], style.thickness), widths[0]);

    // render border
    nvxcu_point4f_t lines[4];

#ifndef USE_GLES
    // HACK:
    // make offsets smaller since antialising helps us to fill corners
    lineWidth -= (lineWidth % 2) ? 1 : 2;
#else
    // HACK:
    // make lineWidth to be odd
    lineWidth -= (lineWidth % 2) ? 0 : 1;
#endif // USE_GLES

    GLfloat thickness2 = lineWidth / 2;
    GLfloat thickness2_ = lineWidth - thickness2;

    // take corners radius into account
    GLfloat maxRadius = std::min(location.end_x - location.start_x,
                                 location.end_y - location.start_y) / 2.0f;

    GLfloat radius = static_cast<GLfloat>(style.radius);

    if (radius > maxRadius)
    {
        NVXIO_PRINT("Max radius (%f) is smaller than specified one (%f). Clamping", maxRadius, radius);
        radius = maxRadius;
    }

    // horizontal lines
    lines[0].x = location.start_x - thickness2 + radius;
    lines[0].y = static_cast<GLfloat>(location.start_y);
    lines[0].z = location.end_x + thickness2_ - radius;
    lines[0].w = static_cast<GLfloat>(location.start_y);

    lines[1].x = location.start_x - thickness2 + radius;
    lines[1].y = static_cast<GLfloat>(location.end_y);
    lines[1].z = location.end_x + thickness2_ - radius;
    lines[1].w = static_cast<GLfloat>(location.end_y);

#ifndef USE_GLES
    // HACK:
    // don't use offsets for OpenGL Full. Antialiasing fills corners for us.
    thickness2 = thickness2_ = 0;
#endif

    // vertical lines
    lines[2].x = static_cast<GLfloat>(location.end_x);
    lines[2].y = location.start_y - thickness2 + radius;
    lines[2].z = static_cast<GLfloat>(location.end_x);
    lines[2].w = location.end_y + thickness2_ - radius;

    lines[3].x = static_cast<GLfloat>(location.start_x);
    lines[3].y = location.start_y - thickness2 + radius;
    lines[3].z = static_cast<GLfloat>(location.start_x);
    lines[3].w = location.end_y + thickness2_ - radius;

    // draw lines

    tmpLinesCPU_.clear();
    for (vx_size i = 0u; i < ovxio::dimOf(lines); ++i)
        tmpLinesCPU_.push_back(lines[i]);

    // draw sectors
    {
        nvxcu_point3f_t c = { location.start_x + radius, location.start_y + radius, radius };
        int num_segments = getNumCircleSegments(c.z);
        genCircleLines(tmpLinesCPU_, c.x, c.y, c.z, num_segments, 4, ovxio::PI_F);

        nvxcu_point3f_t c1 = { location.end_x - radius, location.start_y + radius, radius };
        genCircleLines(tmpLinesCPU_, c1.x, c1.y, c1.z, num_segments, 4, 3.0f * ovxio::PI_F / 2);

        nvxcu_point3f_t c2 = { location.start_x + radius, location.end_y - radius, radius };
        genCircleLines(tmpLinesCPU_, c2.x, c2.y, c2.z, num_segments, 4, ovxio::PI_F / 2);

        nvxcu_point3f_t c3 = { location.end_x - radius, location.end_y - radius, radius };
        genCircleLines(tmpLinesCPU_, c3.x, c3.y, c3.z, num_segments, 4, 0.0f);
    }

    LineStyle lineStyle = {
        { style.color[0], style.color[1], style.color[2], style.color[3] },
        style.thickness
    };

    linesRender_.render(tmpLinesCPU_, lineStyle, textureWidth_, textureHeight_, scaleRatioImage_);

    // render inner region if any
    if (style.isHalfTransparent)
        rectangleRender_.render(location, style, textureWidth_, textureHeight_, scaleRatioImage_);

    // render text if any
    if (!style.label.empty())
    {
        nvxcu_coordinates2d_t textOrigin = {
            location.start_x,
            location.start_y - static_cast<uint32_t>((fontSize + 5) / scaleRatioImage_)
        };

        TextBoxStyle textStyle = {
            {style.color[0], style.color[1], style.color[2], style.color[3]},
            {0u, 0u, 0u, 0u},
            textOrigin
        };

        textRender_.render(style.label, textStyle, textureWidth_, textureHeight_, scaleRatioImage_);
    }
}

void nvidiaio::OpenGLRenderImpl::putArrows(const array_t & old_points, const array_t & new_points,
                                           const LineStyle & line_style)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render2D::putArrows (NVXIO)");

    OpenGLContextSafeSetter setter(holder_);
    arrowsRender_.render(old_points, new_points, line_style, textureWidth_, textureHeight_, scaleRatioImage_);
}

void nvidiaio::OpenGLRenderImpl::putConvexPolygon(const array_t & verticies, const LineStyle& style)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render2D::putConvexPolygon (NVXIO)");

    uint32_t vCount = verticies.num_items;

    NVXIO_ASSERT( verticies.item_type == NVXCU_TYPE_COORDINATES2D );

    if (vCount > 1u)
    {
        Array2CPUPointerMapper mapper(verticies);

        const nvxcu_coordinates2d_t * ptr = static_cast<const nvxcu_coordinates2d_t *>(mapper);

        tmpLinesCPU_.clear();

        for (uint32_t j = 1u; j < vCount; ++j)
        {
            nvxcu_point4f_t line =
            {
                (float)ptr[j-1].x, (float)ptr[j-1].y,
                (float)ptr[j].x, (float)ptr[j].y,
            };

            tmpLinesCPU_.push_back(line);
        }

        // for the last point
        {
            nvxcu_point4f_t line =
            {
                (float)ptr[vCount - 1].x, (float)ptr[vCount - 1].y,
                (float)ptr[0].x, (float)ptr[0].y,
            };

            tmpLinesCPU_.push_back(line);
        }

        OpenGLContextSafeSetter setter(holder_);
        linesRender_.render(tmpLinesCPU_, style, textureWidth_, textureHeight_, scaleRatioImage_);
    }
}

bool nvidiaio::OpenGLRenderImpl::initGL(uint32_t wndWidth, uint32_t wndHeight)
{
    OpenGLContextSafeSetter setter(holder_);

    wndWidth_ = wndWidth;
    wndHeight_ = wndHeight;

    if (!gl_)
    {
        gl_ = std::make_shared<GLFunctions>();

        // requires OpenGL context
        loadGLFunctions(gl_.get());
    }

    // let's use custom framebuffer in video/image mode
    if ((targetType == nvxio::Render::VIDEO_RENDER) ||
            (targetType == nvxio::Render::IMAGE_RENDER))
    {
        gl_->GetIntegerv(GL_FRAMEBUFFER_BINDING, (GLint *)&fboOld_);
        NVXIO_CHECK_GL_ERROR();

        gl_->GenTextures(1, &fboTex_);
        NVXIO_CHECK_GL_ERROR();
        gl_->ActiveTexture(GL_TEXTURE0);
        NVXIO_CHECK_GL_ERROR();
        gl_->BindTexture(GL_TEXTURE_2D, fboTex_);
        NVXIO_CHECK_GL_ERROR();

        if (gl_->IsTexture(fboTex_) == GL_FALSE)
        {
            NVXIO_CHECK_GL_ERROR();
            NVXIO_PRINT("OpenGL render: failed to create texture object");
            return false;
        }

        gl_->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        NVXIO_CHECK_GL_ERROR();
        gl_->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        NVXIO_CHECK_GL_ERROR();
        gl_->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        NVXIO_CHECK_GL_ERROR();
        gl_->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        NVXIO_CHECK_GL_ERROR();
        gl_->TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, wndWidth, wndHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        NVXIO_CHECK_GL_ERROR();

        gl_->GenFramebuffers(1, &fbo_);
        NVXIO_CHECK_GL_ERROR();
        gl_->BindFramebuffer(GL_FRAMEBUFFER, fbo_);
        NVXIO_CHECK_GL_ERROR();
        if (gl_->IsFramebuffer(fbo_) == GL_FALSE)
        {
            NVXIO_CHECK_GL_ERROR();
            NVXIO_PRINT("OpenGL render: failed to create framebuffer object");
            return false;
        }

        gl_->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fboTex_, 0);
        NVXIO_CHECK_GL_ERROR();
        if (gl_->CheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            NVXIO_CHECK_GL_ERROR();
            NVXIO_PRINT("OpenGL render: failed to attach framebuffer");
            return false;
        }
    }

    if (!imageRender_.init(gl_, wndWidth_, wndHeight_))
        return false;

    if (!nv12imageRender_.init(gl_, wndWidth_, wndHeight_))
        return false;

    if (!featuresRender_.init(gl_))
        return false;

    if (!linesRender_.init(gl_))
        return false;

    if (!motionFieldRender_.init(gl_, wndWidth_, wndHeight_))
        return false;

    if (!circlesRender_.init(gl_))
        return false;

    if (!rectangleRender_.init(gl_))
        return false;

    if (!arrowsRender_.init(gl_))
        return false;

    if (!textRender_.init(gl_))
        return false;

    gl_->Enable(GL_BLEND);
    NVXIO_CHECK_GL_ERROR();
    gl_->BlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    NVXIO_CHECK_GL_ERROR();

    return true;
}

void nvidiaio::OpenGLRenderImpl::finalGL()
{
    OpenGLContextSafeSetter setter(holder_);

    imageRender_.release();
    nv12imageRender_.release();
    featuresRender_.release();
    linesRender_.release();
    motionFieldRender_.release();
    circlesRender_.release();
    rectangleRender_.release();
    arrowsRender_.release();
    textRender_.release();

    if (gl_)
    {
        // bind old FBO
        if (fboOld_ != 0u)
        {
            gl_->BindFramebuffer(GL_FRAMEBUFFER, fboOld_);
            NVXIO_CHECK_GL_ERROR();
        }

        if (fbo_ != 0u)
        {
            gl_->DeleteFramebuffers(1, &fbo_);
            NVXIO_CHECK_GL_ERROR();
            fbo_ = 0u;
        }

        if (fboTex_ != 0u)
        {
            gl_->DeleteTextures(1, &fboTex_);
            NVXIO_CHECK_GL_ERROR();
            fboTex_ = 0u;
        }
    }
}

#ifndef __ANDROID__

void nvidiaio::OpenGLRenderImpl::clearGlBuffer()
{
    OpenGLContextSafeSetter setter(holder_);

    gl_->ClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    NVXIO_CHECK_GL_ERROR();
    gl_->Clear(GL_COLOR_BUFFER_BIT);
    NVXIO_CHECK_GL_ERROR();
}

#endif // __ANDROID__

#endif // USE_GUI
