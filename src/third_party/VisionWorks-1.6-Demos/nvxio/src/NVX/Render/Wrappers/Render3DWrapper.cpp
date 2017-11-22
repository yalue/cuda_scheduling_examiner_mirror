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

#include "Render3DWrapper.hpp"
#include "Private/Types.hpp"


#include <cstring>

namespace nvxio
{

Render3DWrapper::Render3DWrapper(std::unique_ptr<nvidiaio::Render3D> render) :
    Render3D(static_cast<TargetType>(render->getTargetType()),
             render->getRenderName()),
    render_(std::move(render)),
    planes_ { },
    points_ { }
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

void Render3DWrapper::putPlanes(const nvxcu_plain_array_t & planes, float * model, const PlaneStyle & style)
{
    // to render planes we must provide data on CPU

    {
        cudaStream_t stream = nullptr;
        uint32_t numItems = 0u;
        size_t itemSize = nvidiaio::getItemSize(planes.base.item_type);

        NVXIO_CUDA_SAFE_CALL( cudaMemcpyAsync(&numItems, planes.num_items_dev_ptr, sizeof(uint32_t),
                                              cudaMemcpyDeviceToHost, stream)  );

        if (planes_.capacity < numItems)
        {
            uint8_t * ptr = static_cast<uint8_t *>(planes_.ptr);
            delete[]ptr;

            planes_.capacity = planes.base.capacity;
            planes_.ptr = new uint8_t[itemSize * planes.base.capacity];
            planes_.item_type = planes.base.item_type;
            planes_.num_items = numItems;
        }

        NVXIO_CUDA_SAFE_CALL( cudaMemcpyAsync(planes_.ptr, planes.dev_ptr,
                                              planes_.num_items * itemSize,
                                              cudaMemcpyDeviceToHost, stream) );

        NVXIO_CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );
    }

    render_->putPlanes(planes_, nvidiaio::matrix4x4f_t(model),
                       *(nvidiaio::Render3D::PlaneStyle *)&style);
}

void Render3DWrapper::putPointCloud(const nvxcu_plain_array_t & points, float * model, const PointCloudStyle & style)
{
    // to render points we must provide data on CPU

    {
        cudaStream_t stream = nullptr;
        uint32_t numItems = 0u;
        size_t itemSize = nvidiaio::getItemSize(points.base.item_type);

        NVXIO_CUDA_SAFE_CALL( cudaMemcpyAsync(&numItems, points.num_items_dev_ptr, sizeof(uint32_t),
                                              cudaMemcpyDeviceToHost, stream)  );

        if (points_.capacity < numItems)
        {
            uint8_t * ptr = static_cast<uint8_t *>(points_.ptr);
            delete[]ptr;

            points_.capacity = points.base.capacity;
            points_.ptr = new uint8_t[itemSize * points.base.capacity];
            points_.item_type = points.base.item_type;
            points_.num_items = numItems;
        }

        NVXIO_CUDA_SAFE_CALL( cudaMemcpyAsync(points_.ptr, points.dev_ptr,
                                              points_.num_items * itemSize,
                                              cudaMemcpyDeviceToHost, stream) );

        NVXIO_CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );
    }

    render_->putPointCloud(points_, nvidiaio::matrix4x4f_t(model),
                           *(nvidiaio::Render3D::PointCloudStyle *)&style);
}

bool Render3DWrapper::flush()
{
    return render_->flush();
}

void Render3DWrapper::close()
{
    if (planes_.ptr)
    {
        uint8_t * ptr = static_cast<uint8_t *>(planes_.ptr);
        delete[]ptr;
        planes_.ptr = nullptr;
    }

    if (points_.ptr)
    {
        uint8_t * ptr = static_cast<uint8_t *>(points_.ptr);
        delete[]ptr;
        points_.ptr = nullptr;
    }

    render_->close();
}

void Render3DWrapper::setViewMatrix(float * view)
{
    render_->setViewMatrix(nvidiaio::matrix4x4f_t(view));
}

void Render3DWrapper::getViewMatrix(float * view) const
{
    nvidiaio::matrix4x4f_t wrapper(view);
    render_->getViewMatrix(wrapper);
}

void Render3DWrapper::setProjectionMatrix(float * projection)
{
    render_->setProjectionMatrix(nvidiaio::matrix4x4f_t(projection));
}

void Render3DWrapper::getProjectionMatrix(float * projection) const
{
    nvidiaio::matrix4x4f_t wrapper(projection);
    render_->getProjectionMatrix(wrapper);
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

uint32_t Render3DWrapper::getWidth() const
{
    return render_->getWidth();
}

uint32_t Render3DWrapper::getHeight() const
{
    return render_->getHeight();
}

void Render3DWrapper::putImage(const nvxcu_pitch_linear_image_t & image)
{
    render_->putImage(nvidiaio::image_t(image));
}

void Render3DWrapper::putText(const std::string& text, const nvxio::Render::TextBoxStyle& style)
{
    render_->putText(text, style);
}

Render3DWrapper::~Render3DWrapper()
{
    close();
}

} // namespace nvxio
