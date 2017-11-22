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

#ifndef NVIDIAIO_RENDER_HPP
#define NVIDIAIO_RENDER_HPP

#include <memory>
#include <string>

#include "Private/Types.hpp"
#include <NVX/Render.hpp>

#ifndef __ANDROID__
# include <NVX/Application.hpp>
#endif

namespace nvidiaio
{

class Render
{
public:

#ifndef __ANDROID__
    typedef nvxio::Render::OnKeyboardEventCallback OnKeyboardEventCallback;
    typedef nvxio::Render::OnMouseEventCallback OnMouseEventCallback;
#endif

    typedef nvxio::Render::TextBoxStyle TextBoxStyle;
    typedef nvxio::Render::FeatureStyle FeatureStyle;
    typedef nvxio::Render::LineStyle LineStyle;
    typedef nvxio::Render::MotionFieldStyle MotionFieldStyle;
    typedef nvxio::Render::DetectedObjectStyle DetectedObjectStyle;
    typedef nvxio::Render::CircleStyle CircleStyle;

#ifndef __ANDROID__
    typedef nvxio::Render::MouseButtonEvent MouseButtonEvent;
#endif
    typedef nvxio::Render::TargetType TargetType;

#ifndef __ANDROID__
    virtual void setOnKeyboardEventCallback(OnKeyboardEventCallback callback, void * context) = 0;
    virtual void setOnMouseEventCallback(OnMouseEventCallback callback, void * context) = 0;
#endif

    virtual void putImage(const image_t & image) = 0;
    virtual void putTextViewport(const std::string& text, const TextBoxStyle& style) = 0;
    virtual void putFeatures(const array_t & location, const FeatureStyle& style) = 0;
    virtual void putFeatures(const array_t & location, const array_t & styles) = 0;
    virtual void putLines(const array_t & lines, const LineStyle& style) = 0;
    virtual void putConvexPolygon(const array_t & vertices, const LineStyle& style) = 0;
    virtual void putMotionField(const image_t & field, const MotionFieldStyle& style) = 0;
    virtual void putObjectLocation(const nvxcu_rectangle_t & location, const DetectedObjectStyle& style) = 0;
    virtual void putCircles(const array_t & circles, const CircleStyle& style) = 0;
    virtual void putArrows(const array_t & old_points, const array_t & new_points,
                           const LineStyle& style) = 0;

#ifndef __ANDROID__
    virtual bool flush() = 0;
    virtual void close() = 0;
#endif

    TargetType getTargetType() const
    {
        return targetType;
    }

    std::string getRenderName() const
    {
        return renderName;
    }

    virtual uint32_t getViewportWidth() const = 0;
    virtual uint32_t getViewportHeight() const = 0;

    virtual ~Render()
    {}

protected:

    Render(TargetType type = nvxio::Render::UNKNOWN_RENDER, std::string name = "Undefined"):
        targetType(type),
        renderName(name)
    {}

    const TargetType  targetType;
    const std::string renderName;
};

#ifdef __ANDROID__

std::unique_ptr<Render> createRender(uint32_t width, uint32_t height, bool doScale = true);

#else

std::unique_ptr<Render> createDefaultRender(const std::string& title, uint32_t width, uint32_t height,
                                            nvxcu_df_image_e format = NVXCU_DF_IMAGE_RGBX, bool doScale = true,
                                            bool fullScreen = nvxio::Application::get().getFullScreenFlag());

std::unique_ptr<Render> createVideoRender(const std::string& path, uint32_t width,
                                          uint32_t height, nvxcu_df_image_e format = NVXCU_DF_IMAGE_RGBX);

std::unique_ptr<Render> createWindowRender(const std::string& title, uint32_t width, uint32_t height,
                                           nvxcu_df_image_e format = NVXCU_DF_IMAGE_RGBX, bool doScale = true,
                                           bool fullscreen = nvxio::Application::get().getFullScreenFlag());

std::unique_ptr<Render> createImageRender(const std::string& path, uint32_t width, uint32_t height,
                                          nvxcu_df_image_e format = NVXCU_DF_IMAGE_RGBX);

#endif

} // namespace nvidiaio

#endif // NVIDIAIO_RENDER_HPP
