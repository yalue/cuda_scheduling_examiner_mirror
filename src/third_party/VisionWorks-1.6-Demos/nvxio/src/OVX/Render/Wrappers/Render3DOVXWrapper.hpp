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

#ifndef RENDER3D_WRAPPER_HPP
#define RENDER3D_WRAPPER_HPP

#include <memory>

#include <OVX/Render3DOVX.hpp>

#include "Render/Render3DImpl.hpp"
#include "../../Private/TypesOVX.hpp"

namespace ovxio
{

class Render3DWrapper :
        public Render3D
{
public:
    Render3DWrapper(vx_context context, std::unique_ptr<nvidiaio::Render3D> render);

    virtual void setOnKeyboardEventCallback(OnKeyboardEventCallback callback, void * context);
    virtual void setOnMouseEventCallback(OnMouseEventCallback callback, void * context);

    virtual void putPlanes(vx_array planes, vx_matrix model, const PlaneStyle& style);
    virtual void putPointCloud(vx_array points, vx_matrix model, const PointCloudStyle& style);

    virtual bool flush();
    virtual void close();

    virtual void setViewMatrix(vx_matrix view);
    virtual void getViewMatrix(vx_matrix view) const;

    virtual void setProjectionMatrix(vx_matrix projection);
    virtual void getProjectionMatrix(vx_matrix projection) const;

    virtual void setDefaultFOV(float fov);

    virtual void enableDefaultKeyboardEventCallback();
    virtual void disableDefaultKeyboardEventCallback();

    virtual bool useDefaultKeyboardEventCallback();

    virtual vx_uint32 getWidth() const;
    virtual vx_uint32 getHeight() const;

    virtual void putImage(vx_image image);

    virtual void putText(const std::string& text, const ovxio::Render::TextBoxStyle& style);

    virtual ~Render3DWrapper();

private:
    vx_context context_;
    std::unique_ptr<nvidiaio::Render3D> render_;
};

} // namespace ovxio

#endif // RENDER3D_WRAPPER_HPP
