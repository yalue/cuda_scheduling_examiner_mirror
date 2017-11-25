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

#include "Private/Types.hpp"
#include "TypesOVX.hpp"

#include <cuda_runtime_api.h>
#include <cstring>

namespace ovxio
{

image_t::image_t(vx_image image, vx_enum usage, vx_enum mem_type) :
    nvidiaio::image_t(),
    // OpenVX
    image_ (image),
    addrs_ { },
    ptrs_ { },
    map_ids_ { }

{
    vx_df_image_e vxFormat = VX_DF_IMAGE_VIRT;

    NVXIO_SAFE_CALL( vxQueryImage(image_, VX_IMAGE_ATTRIBUTE_PLANES, (void *)&planes_, sizeof(planes_)) );
    NVXIO_SAFE_CALL( vxQueryImage(image_, VX_IMAGE_ATTRIBUTE_FORMAT, (void *)&vxFormat, sizeof(vxFormat)) );
    NVXIO_SAFE_CALL( vxQueryImage(image_, VX_IMAGE_ATTRIBUTE_WIDTH, (void *)&width, sizeof(width)) );
    NVXIO_SAFE_CALL( vxQueryImage(image_, VX_IMAGE_ATTRIBUTE_HEIGHT, (void *)&height, sizeof(height)) );

    format = static_cast<nvxcu_df_image_e>(vxFormat);

    for (vx_uint32 p = 0u; p < planes_; ++p)
    {
        NVXIO_SAFE_CALL( vxMapImagePatch(image_, nullptr, p, map_ids_ + p, addrs_ + p, ptrs_ + p, usage, mem_type, 0) );

        planes[p].ptr = ptrs_[p];
        planes[p].pitch_in_bytes = addrs_[p].stride_y;
    }
}

image_t::~image_t()
{
    if (image_)
    {
        for (vx_uint32 p = 0u; p < planes_; ++p)
        {
            vxUnmapImagePatch(image_, map_ids_[p]);
        }
    }
}

array_t::array_t(vx_array array, vx_enum usage, vx_enum mem_type) :
    nvidiaio::array_t(),
    // OpenVX stuff
    array_(array),
    map_id(0ul),
    size_(0ul)
{
    vx_size vxCapacity = 0ul, stride = 0ul;
    vx_enum vxItemType = 0;

    NVXIO_SAFE_CALL( vxQueryArray(array_, VX_ARRAY_ATTRIBUTE_NUMITEMS, &size_, sizeof(size_)) );
    NVXIO_SAFE_CALL( vxQueryArray(array_, VX_ARRAY_ATTRIBUTE_CAPACITY, &vxCapacity, sizeof(vxCapacity)) );
    NVXIO_SAFE_CALL( vxQueryArray(array_, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &vxItemType, sizeof(vxItemType)) );

    if (size_ > 0)
        NVXIO_SAFE_CALL( vxMapArrayRange(array_, 0, size_, &map_id, &stride, &ptr, usage, mem_type, 0) );

    capacity = static_cast<uint32_t>(capacity);
    num_items = static_cast<uint32_t>(size_);
    item_type = static_cast<nvxcu_array_item_type_e>(vxItemType);
}

array_t::~array_t()
{
    if (array_)
    {
        if (size_ > 0)
            vxUnmapArrayRange(array_, map_id);
    }
}

void matrix4x4f_t::assert4x4f(vx_matrix matrix)
{
    vx_enum type = 0;
    vx_size rows = 0ul, cols = 0ul;

    NVXIO_SAFE_CALL( vxQueryMatrix(matrix, VX_MATRIX_ATTRIBUTE_TYPE, &type, sizeof(type)) );
    NVXIO_ASSERT(type == VX_TYPE_FLOAT32);

    NVXIO_SAFE_CALL( vxQueryMatrix(matrix, VX_MATRIX_ATTRIBUTE_ROWS, &rows, sizeof(rows)) );
    NVXIO_SAFE_CALL( vxQueryMatrix(matrix, VX_MATRIX_ATTRIBUTE_COLUMNS, &cols, sizeof(cols)) );
    NVXIO_ASSERT(rows == 4 && cols == 4);
}

matrix4x4f_t::matrix4x4f_t(vx_matrix array) :
    nvidiaio::matrix4x4f_t()
{
    assert4x4f(array);

    NVXIO_SAFE_CALL( vxReadMatrix(array, (void *)ptr) );
}

}
