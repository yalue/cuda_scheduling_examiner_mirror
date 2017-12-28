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

#ifndef NVXCUIO_UTILITY_HPP
#define NVXCUIO_UTILITY_HPP

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <string>

#ifdef __ANDROID__
#include <android/log.h>
#endif

#include <NVX/nvx.h>
#include <NVX/Export.hpp>

/**
 * \file
 * \brief The `NVXCUIO` utility functions.
 */

#define NVXIO_THROW_EXCEPTION(msg) \
    do { \
        std::ostringstream ostr_; \
        ostr_ << msg; \
        throw std::runtime_error(ostr_.str()); \
    } while(0)

#define NVXIO_ASSERT(cond) \
    do \
    { \
        if (!(cond)) \
        { \
            NVXIO_THROW_EXCEPTION(#cond << " failure in file " << __FILE__ << " line " << __LINE__); \
        } \
    } while (0)
#define THROW_EXCEPTION(msg) \
    do { \
        std::ostringstream ostr_; \
        ostr_ << msg; \
        throw std::runtime_error(ostr_.str()); \
    } while(0)

#define ASSERT(cond) \
    do \
    { \
        bool stat = (cond); \
        if (!stat) \
        { \
            THROW_EXCEPTION(#cond << " failure in file " << __FILE__ << " line " << __LINE__); \
        } \
    } while (0)

#define NVXCU_SAFE_CALL(nvxcuOp) \
    do \
    { \
        nvxcu_error_status_e stat = (nvxcuOp); \
        if (stat != NVXCU_SUCCESS) \
        { \
            THROW_EXCEPTION(#nvxcuOp << " failure [status = " << stat << "]" << " in file " << __FILE__ << " line " << __LINE__); \
        } \
    } while (0)

#define CUDA_SAFE_CALL(cudaOp) \
    do \
    { \
        cudaError_t err = (cudaOp); \
        if (err != cudaSuccess) \
        { \
            THROW_EXCEPTION(#cudaOp << " failure [CUDA error = " << err << "]" << " in file " << __FILE__ << " line " << __LINE__);  \
        } \
    } while (0)

#define NVXIO_CUDA_SAFE_CALL(cudaOp) \
    do \
    { \
        cudaError_t err = (cudaOp); \
        if (err != cudaSuccess) \
        { \
            std::ostringstream ostr; \
            ostr << "CUDA Error in " << #cudaOp << __FILE__ << " file " << __LINE__ << " line : " << cudaGetErrorString(err); \
            throw std::runtime_error(ostr.str()); \
        } \
    } while (0)


namespace nvxio
{
/**
 * \ingroup group_nvxcuio_utility
 * \brief make_unique function.
 * \see nvx_nvxcuio_api
 */
template <typename T, typename... Args>
std::unique_ptr<T> makeUP(Args &&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

#ifndef __ANDROID__

/**
 * \ingroup group_nvxio_utility
 * \brief Returns a vector of NVXIO supported features.
 *
 * The following scheme is used to describe the features in the form of tree:
 *
 *       <feature_type>:<IO_object_type>:<backend>:<details_0>:<details_1>:...
 *
 * `<feature_type>` possible values:
 * - `render2d` - describes a set of 2D renders.
 * - `render3d` - describes a set of 3D renders.
 * - `source` - describes a set of frame sources.
 *
 * `<IO_object_type>` possible values:
 *
 * - `image` - a sequence of images. The `source` features read the sequence; the
 *   `render` features write the sequence.
 * - `video` - a video file.
 * - `window` - a UI window.
 * - `camera` - different types of cameras - USB, CSI, etc.
 *
 * `<backend>` possible values:
 * - `opencv` - implementation using OpenCV drawing utilities.
 * - `opengl` - implementation using OpenGL (ES) shaders.
 * - `gstreamer` -  GStreamer implementation.
 * - `v4l2` - Video 4 Linux 2 implementation
 * - `nvmedia` - NvMedia implementation
 * - `openmax` - OpenMAX implementation
 *
 * `<details_N>` tags can store any additional information about the feature.
 *
 * Examples:
 *
 * - `render2d:video:gstreamer`: a 2D render that can write to a video file using
 *   GStreamer backend
 *
 * - `source:video:nvmedia:pure`: a frame source that can fetch images from a video
 *   file using the "pure" NvMedia backend
 *
 * - `source:camera:nvmedia:pure:dvp-ov10635-yuv422-ab`: a frame source that can
 *   fetch images from an OV10635 camera attached to `ab` ports in a YUV422 format
 *   using "pure" NvMedia backend.
 *
 * \see nvx_nvxio_api
 */
NVXIO_EXPORT std::vector<std::string> getSupportedFeatures();

#endif

} // namespace nvxio

#endif // NVXCUIO_UTILITY_HPP
