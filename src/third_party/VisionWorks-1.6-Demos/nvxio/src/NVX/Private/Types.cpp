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

#include <OVX/UtilityOVX.hpp>

#include "Types.hpp"

#include <cuda_runtime_api.h>
#include <cstring>

namespace nvidiaio
{

size_t getItemSize(nvxcu_array_item_type_e item_type)
{
    if (item_type == NVXCU_TYPE_RECTANGLE)
        return sizeof(nvxcu_rectangle_t);
    if (item_type == NVXCU_TYPE_KEYPOINT)
        return sizeof(nvxcu_keypoint_t);
    if (item_type == NVXCU_TYPE_COORDINATES2D)
        return sizeof(nvxcu_coordinates2d_t);
    if (item_type == NVXCU_TYPE_COORDINATES3D)
        return sizeof(nvxcu_coordinates3d_t);

    if (item_type == NVXCU_TYPE_POINT2F)
        return sizeof(nvxcu_point2f_t);
    if (item_type == NVXCU_TYPE_POINT3F)
        return sizeof(nvxcu_point3f_t);
    if (item_type == NVXCU_TYPE_POINT4F)
        return sizeof(nvxcu_point4f_t);
    if (item_type == NVXCU_TYPE_KEYPOINTF)
        return sizeof(nvxcu_keypointf_t);

    return 0ul;
}

//
// Image
//

image_t::image_t() :
    format(),
    width(0u),
    height(0u),
    planes_(0ul),
    planes { }
{

}

image_t::image_t(const nvxcu_pitch_linear_image_t & image) :
    format(image.base.format),
    width(image.base.width),
    height(image.base.height),
    planes_(),
    planes { }
{
    planes_ = 1u;

    NVXIO_ASSERT(format == NVXCU_DF_IMAGE_U8 ||
                 format == NVXCU_DF_IMAGE_RGB ||
                 format == NVXCU_DF_IMAGE_RGBX ||
                 format == NVXCU_DF_IMAGE_NV12);

    NVXIO_ASSERT(image.base.image_type == NVXCU_PITCH_LINEAR_IMAGE);

    if (format == NVXCU_DF_IMAGE_NV12)
        planes_ = 2u;

    for (uint32_t p = 0u; p < planes_; ++p)
    {
        planes[p].ptr = image.planes[p].dev_ptr;
        planes[p].pitch_in_bytes = image.planes[p].pitch_in_bytes;
    }
}

//
// Array
//

array_t::array_t(const nvxcu_plain_array_t & array) :
    item_type(array.base.item_type),
    ptr(array.dev_ptr),
    num_items(0u),
    capacity(array.base.capacity)
{
    NVXIO_ASSERT(array.base.array_type == NVXCU_PLAIN_ARRAY);

    cudaStream_t stream = nullptr;

    NVXIO_CUDA_SAFE_CALL( cudaMemcpyAsync(&num_items, array.num_items_dev_ptr,
                                          sizeof(num_items),
                                          cudaMemcpyDeviceToHost, stream) );

    NVXIO_CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );
}

array_t::array_t() :
    item_type(),
    ptr(nullptr),
    num_items(0u),
    capacity(0u)
{

}

//
// Mappers
//

Array2CPUPointerMapper::Array2CPUPointerMapper(const array_t & array, std::vector<uint8_t> * cpuData) :
    cpuData_ { },
    cpuDataPointer_(nullptr)
{
    size_t size = array.num_items * getItemSize(array.item_type);
    NVXIO_ASSERT(size > 0ul);

    std::vector<uint8_t> & vec = cpuData ? *cpuData : cpuData_;
    vec.resize(size);

    cpuDataPointer_ = &vec[0];

    cudaStream_t stream = nullptr;
    NVXIO_CUDA_SAFE_CALL( cudaMemcpyAsync(cpuDataPointer_, array.ptr, size,
                                          cudaMemcpyDeviceToHost, stream) );
    NVXIO_CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );
}

Image2CPUPointerMapper::Image2CPUPointerMapper(const image_t & image) :
    cpuData_ { }
{
    NVXIO_ASSERT(image.format == NVXCU_DF_IMAGE_U8 ||
                 image.format == NVXCU_DF_IMAGE_RGB ||
                 image.format == NVXCU_DF_IMAGE_RGBX ||
                 image.format == NVXCU_DF_IMAGE_2F32);

    size_t pitch = image.width *
            (image.format == NVXCU_DF_IMAGE_U8 ? sizeof(uint8_t) :
             image.format == NVXCU_DF_IMAGE_RGB ? 3 * sizeof(uint8_t) :
             image.format == NVXCU_DF_IMAGE_RGBX ? 4 * sizeof(uint8_t) :
             image.format == NVXCU_DF_IMAGE_2F32 ? 2 * sizeof(float) : 0ul);
    size_t size = pitch * image.height;
    NVXIO_ASSERT(size > 0ul);

    cpuData_.resize(size);

    cudaStream_t stream = nullptr;
    NVXIO_CUDA_SAFE_CALL( cudaMemcpy2DAsync(&cpuData_[0], pitch,
                                            image.planes[0].ptr, image.planes[0].pitch_in_bytes,
                                            pitch, image.height,
                                            cudaMemcpyDeviceToHost, stream) );
    NVXIO_CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );
}

//
// Matrix4x4f
//

matrix4x4f_t::matrix4x4f_t() :
    ptr(storage_), storage_{ }
{
}

matrix4x4f_t::matrix4x4f_t(const matrix4x4f_t & array) :
    ptr(storage_), storage_{ }
{
    std::memcpy(ptr, array.ptr, sizeof(array.storage_));
}

matrix4x4f_t::matrix4x4f_t(float * array) :
    ptr(array), storage_{ }
{
}


matrix4x4f_t & matrix4x4f_t::operator = (const matrix4x4f_t & array)
{
    if (this != &array)
    {
        std::memset(storage_, 0, sizeof(storage_));
        std::memcpy(ptr, array.ptr, sizeof(array.storage_));
    }

    return *this;
}

}
