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

#ifndef TYPES_HPP
#define TYPES_HPP

#include <cstdint>
#include <vector>

#include <NVX/nvxcu.h>
#include <VX/vx.h>

namespace nvidiaio
{

#define NVXCU_DF_IMAGE_NONE static_cast<nvxcu_df_image_e>(0)

size_t getItemSize(nvxcu_array_item_type_e item_type);

//----------------------------------------------------------------------------
// Image wrapper
//----------------------------------------------------------------------------

#define NVIDIAIO_NB_MAX_PLANES     (4u)

class image_t
{
public:
    nvxcu_df_image_e format;
    uint32_t width;
    uint32_t height;
    size_t planes_;

    struct
    {
        void * ptr;
        int32_t pitch_in_bytes;
    } planes[NVIDIAIO_NB_MAX_PLANES];

    explicit image_t(const nvxcu_pitch_linear_image_t & image);
    image_t();

    template <typename T>
    operator const T * () const;

//    ~image_t();

private:

    image_t(const image_t &) = delete;
    image_t & operator = (const image_t &) = delete;

};

class Image2CPUPointerMapper
{
public:
    explicit Image2CPUPointerMapper(const image_t & image);

    template <typename T>
    operator const T * () const;

private:
    Image2CPUPointerMapper(const Image2CPUPointerMapper &) = delete;
    const Image2CPUPointerMapper & operator= (const Image2CPUPointerMapper &) = delete;

    std::vector<uint8_t> cpuData_;
};

template <typename T>
Image2CPUPointerMapper::operator const T * () const
{
    return (const T *)&cpuData_[0];
}

//----------------------------------------------------------------------------
// Array wrapper
//----------------------------------------------------------------------------

class array_t
{
public:
    nvxcu_array_item_type_e item_type;
    void * ptr;
    uint32_t num_items;
    uint32_t capacity;

    array_t();
    explicit array_t(const nvxcu_plain_array_t & array);

private:

    array_t(const array_t &) = delete;
    array_t & operator = (const array_t &) = delete;

};

class Array2CPUPointerMapper
{
public:
    explicit Array2CPUPointerMapper(const array_t & array, std::vector<uint8_t> * cpuData = nullptr);

    template <typename T>
    operator const T * () const;

private:
    Array2CPUPointerMapper(const Array2CPUPointerMapper &) = delete;
    const Array2CPUPointerMapper & operator= (const Array2CPUPointerMapper &) = delete;

    std::vector<uint8_t> cpuData_;
    uint8_t * cpuDataPointer_;
};

template <typename T>
Array2CPUPointerMapper::operator const T * () const
{
    return (const T *)cpuDataPointer_;
}

//----------------------------------------------------------------------------
// Matrix wrapper
//----------------------------------------------------------------------------

class matrix4x4f_t
{
public:
    float * ptr;

    matrix4x4f_t();
    explicit matrix4x4f_t(const matrix4x4f_t & array);
    explicit matrix4x4f_t(float * array);
    matrix4x4f_t & operator = (const matrix4x4f_t &);

private:

    // internal storage
    float storage_[4 * 4];
};

} // namespace nvidiaio


namespace nvxio
{

class image_t : public nvidiaio::image_t
{
};

class matrix4x4f_t : public nvidiaio::matrix4x4f_t
{
};

}


#endif // TYPES_HPP
