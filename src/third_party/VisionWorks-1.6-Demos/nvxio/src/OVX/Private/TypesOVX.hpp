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

#ifndef TYPESNVXIO_HPP
#define TYPESNVXIO_HPP

#include <cstdint>
#include <vector>

#include <NVX/nvxcu.h>
#include <VX/vx.h>

namespace ovxio
{
class image_t: public nvidiaio::image_t
{
public:
    image_t(vx_image image, vx_enum usage, vx_enum mem_type);
    ~image_t();
private:

    // OpenVX stuff
    vx_image image_;

    vx_imagepatch_addressing_t addrs_[NVIDIAIO_NB_MAX_PLANES];
    void * ptrs_[NVIDIAIO_NB_MAX_PLANES];
    vx_map_id map_ids_[NVIDIAIO_NB_MAX_PLANES];
};

class array_t: public nvidiaio::array_t
{
public:
    array_t(vx_array array, vx_enum usage, vx_enum mem_type);
    ~array_t();

private:
    array_t(const array_t &) = delete;
    array_t & operator = (const array_t &) = delete;

private:

    // OpenVX stuff
    vx_array array_;
    vx_map_id map_id;
    vx_size size_;
};

class matrix4x4f_t: public nvidiaio::matrix4x4f_t
{
public:

    explicit matrix4x4f_t(vx_matrix array);
    static void assert4x4f(vx_matrix matrix);
};

}

#endif // TYPESNVXIO_HPP
