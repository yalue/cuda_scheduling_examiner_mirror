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

#ifndef NVIDIAIO_RENDER3D_HPP
#define NVIDIAIO_RENDER3D_HPP

#include <memory>
#include <string>

#include "Private/Types.hpp"
#include "Render/RenderImpl.hpp"

#include <NVX/Render3D.hpp>

namespace nvidiaio
{

class Render3D
{
public:
    typedef nvxio::Render3D::TargetType TargetType;
    typedef nvxio::Render3D::MouseButtonEvent MouseButtonEvent;

    typedef nvxio::Render3D::PlaneStyle PlaneStyle;
    typedef nvxio::Render3D::PointCloudStyle PointCloudStyle;

    typedef nvxio::Render3D::OnKeyboardEventCallback OnKeyboardEventCallback;
    typedef nvxio::Render3D::OnMouseEventCallback OnMouseEventCallback;

    typedef nvidiaio::Render::TextBoxStyle TextBoxStyle;

    virtual void setOnKeyboardEventCallback(OnKeyboardEventCallback callback, void * context) = 0;
    virtual void setOnMouseEventCallback(OnMouseEventCallback callback, void * context) = 0;

    virtual void putPlanes(const array_t & planes, const matrix4x4f_t & model, const PlaneStyle & style) = 0;
    virtual void putPointCloud(const array_t & points, const matrix4x4f_t & model, const PointCloudStyle & style) = 0;
    virtual void putImage(const image_t & image) = 0;
    virtual void putText(const std::string& text, const TextBoxStyle & style) = 0;

    virtual bool flush() = 0;
    virtual void close() = 0;

    virtual void setViewMatrix(const matrix4x4f_t & view) = 0;
    virtual void getViewMatrix(matrix4x4f_t & view) const = 0;

    virtual void setProjectionMatrix(const matrix4x4f_t & projection) = 0;
    virtual void getProjectionMatrix(matrix4x4f_t & projection) const = 0;

    virtual void setDefaultFOV(float fov) = 0;
    virtual void enableDefaultKeyboardEventCallback() = 0;

    virtual void disableDefaultKeyboardEventCallback() = 0;
    virtual bool useDefaultKeyboardEventCallback() = 0;

    virtual uint32_t getWidth() const = 0;
    virtual uint32_t getHeight() const = 0;

    TargetType getTargetType() const
    {
        return targetType;
    }

    std::string getRenderName() const
    {
        return renderName;
    }

    virtual ~Render3D()
    { }

protected:
    Render3D(TargetType type = nvxio::Render3D::UNKNOWN_RENDER, const std::string& name = "Undefined"):
        targetType(type),
        renderName(name)
    { }

    const TargetType  targetType;
    const std::string renderName;
};

std::unique_ptr<Render3D> createDefaultRender3D(int32_t xPos, int32_t yPos, const std::string& title,
                                                uint32_t width, uint32_t height);

} // namespace nvidiaio

#endif // NVIDIAIO_RENDER3D_HPP
