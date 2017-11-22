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

#include "Render3DOVXWrapper.hpp"
#include "Private/Types.hpp"

#include <OVX/UtilityOVX.hpp>

#include <cstring>

namespace ovxio
{

Render3DWrapper::Render3DWrapper(vx_context context, std::unique_ptr<nvidiaio::Render3D> render) :
    Render3D(static_cast<TargetType>(render->getTargetType()),
             render->getRenderName()),
    context_(context),
    render_(std::move(render))
{
}

void Render3DWrapper::setOnKeyboardEventCallback(OnKeyboardEventCallback callback, void * context)
{
    render_->setOnKeyboardEventCallback(reinterpret_cast<nvidiaio::Render3D::OnKeyboardEventCallback>(callback),
                                        context);
}

void Render3DWrapper::setOnMouseEventCallback(OnMouseEventCallback callback, void * context)
{
    render_->setOnMouseEventCallback(reinterpret_cast<nvidiaio::Render3D::OnMouseEventCallback>(callback),
                                     context);
}

void Render3DWrapper::putPlanes(vx_array planes, vx_matrix model, const PlaneStyle& style)
{
    render_->putPlanes(ovxio::array_t(planes, VX_READ_ONLY, VX_MEMORY_TYPE_HOST), ovxio::matrix4x4f_t(model),
                       *(nvidiaio::Render3D::PlaneStyle *)&style);
}

void Render3DWrapper::putPointCloud(vx_array points, vx_matrix model, const PointCloudStyle& style)
{
    render_->putPointCloud(ovxio::array_t(points, VX_READ_ONLY, VX_MEMORY_TYPE_HOST), ovxio::matrix4x4f_t(model),
                           *(nvidiaio::Render3D::PointCloudStyle *)&style);
}

bool Render3DWrapper::flush()
{
    return render_->flush();
}

void Render3DWrapper::close()
{
    render_->close();
}

void Render3DWrapper::setViewMatrix(vx_matrix view)
{
    render_->setViewMatrix(ovxio::matrix4x4f_t(view));
}

void Render3DWrapper::getViewMatrix(vx_matrix view) const
{
    nvidiaio::matrix4x4f_t mat;
    render_->getViewMatrix(mat);

    ovxio::matrix4x4f_t::assert4x4f(view);
    NVXIO_SAFE_CALL( vxWriteMatrix(view, (void *)mat.ptr) );
}

void Render3DWrapper::setProjectionMatrix(vx_matrix projection)
{
    render_->setProjectionMatrix(ovxio::matrix4x4f_t(projection));
}

void Render3DWrapper::getProjectionMatrix(vx_matrix projection) const
{
    nvidiaio::matrix4x4f_t mat;
    render_->getProjectionMatrix(mat);

    ovxio::matrix4x4f_t::assert4x4f(projection);
    NVXIO_SAFE_CALL( vxWriteMatrix(projection, (void *)mat.ptr) );
}

void Render3DWrapper::setDefaultFOV(float fov)
{
    render_->setDefaultFOV(fov);
}

void Render3DWrapper::enableDefaultKeyboardEventCallback()
{
    render_->enableDefaultKeyboardEventCallback();
}

void Render3DWrapper::disableDefaultKeyboardEventCallback()
{
    render_->disableDefaultKeyboardEventCallback();
}

bool Render3DWrapper::useDefaultKeyboardEventCallback()
{
    return render_->useDefaultKeyboardEventCallback();
}

vx_uint32 Render3DWrapper::getWidth() const
{
    return render_->getWidth();
}

vx_uint32 Render3DWrapper::getHeight() const
{

    return render_->getHeight();
}

void Render3DWrapper::putImage(vx_image image)
{
    render_->putImage(ovxio::image_t(image, VX_READ_ONLY, NVX_MEMORY_TYPE_CUDA));
}

void Render3DWrapper::putText(const std::string& text, const ovxio::Render::TextBoxStyle & style)
{
    render_->putText(text,
                     *(nvidiaio::Render::TextBoxStyle *)&style);
}

Render3DWrapper::~Render3DWrapper()
{
    close();
}

} // namespace ovxio
