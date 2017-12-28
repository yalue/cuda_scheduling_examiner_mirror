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

#ifndef NVXIO_UTILITY_HPP
#define NVXIO_UTILITY_HPP

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
 * \brief The `NVXIO` utility functions.
 */

namespace ovxio
{

/**
 * \defgroup group_nvxio_utility Utility
 * \ingroup nvx_nvxio_api
 *
 * Defines NVXIO Utility API.
 */

//
// NVXIO_LOG* macros
//

#ifdef __ANDROID__

/**
 * \ingroup group_nvxio_utility
 * \brief Adds android log entry with VERBOSE level.
 * \see nvx_nvxio_api
 */
#define NVXIO_LOGV(tag, ...) ((void)__android_log_print(ANDROID_LOG_VERBOSE, tag, __VA_ARGS__))

/**
 * \ingroup group_nvxio_utility
 * \brief Adds android log entry with DEBUG level.
 * \see nvx_nvxio_api
 */
#define NVXIO_LOGD(tag, ...) ((void)__android_log_print(ANDROID_LOG_DEBUG, tag, __VA_ARGS__))

/**
 * \ingroup group_nvxio_utility
 * \brief Adds android log entry with INFO level.
 * \see nvx_nvxio_api
 */
#define NVXIO_LOGI(tag, ...) ((void)__android_log_print(ANDROID_LOG_INFO, tag, __VA_ARGS__))

/**
 * \ingroup group_nvxio_utility
 * \brief Adds android log entry with WARN level.
 * \see nvx_nvxio_api
 */
#define NVXIO_LOGW(tag, ...) ((void)__android_log_print(ANDROID_LOG_WARN, tag, __VA_ARGS__))

/**
 * \ingroup group_nvxio_utility
 * \brief Adds android log entry with ERROR level.
 * \see nvx_nvxio_api
 */
#define NVXIO_LOGE(tag, ...) ((void)__android_log_print(ANDROID_LOG_ERROR, tag, __VA_ARGS__))

#endif

//
// Auxiliary macros
//

/**
 * \ingroup group_nvxio_utility
 * \brief Throws `std::runtime_error` exception.
 * \param [in] msg A message with content related to the exception.
 * \see nvx_nvxio_api
 */
#define NVXIO_THROW_EXCEPTION(msg) \
    do { \
        std::ostringstream ostr_; \
        ostr_ << msg; \
        throw std::runtime_error(ostr_.str()); \
    } while(0)

/**
 * \ingroup group_nvxio_utility
 * \brief Performs an NVX operation. If the operation fails, then it throws `std::runtime_error` exception.
 * \param [in] vxOp A function to be called.
 * The function must have `vx_status` return value.
 * \see nvx_nvxio_api
 */
#define NVXIO_SAFE_CALL(vxOp) \
    do \
    { \
        vx_status status = (vxOp); \
        if (status != VX_SUCCESS) \
        { \
            NVXIO_THROW_EXCEPTION(#vxOp << " failure [status = " << status << "]" << " in file " << __FILE__ << " line " << __LINE__); \
        } \
    } while (0)

/**
 * \ingroup group_nvxio_utility
 * \brief Checks a condition. If the condition is false then it throws `std::runtime_error` exception.
 * \param [in] cond Expression to be evaluated.
 * \see nvx_nvxio_api
 */
#define NVXIO_ASSERT(cond) \
    do \
    { \
        if (!(cond)) \
        { \
            NVXIO_THROW_EXCEPTION(#cond << " failure in file " << __FILE__ << " line " << __LINE__); \
        } \
    } while (0)

/**
 * \ingroup group_nvxio_utility
 * \brief Checks a reference. If the reference is not valid then it throws `std::runtime_error` exception.
 * \param [in] ref Reference to be checked.
 * \see nvx_nvxio_api
 */
#define NVXIO_CHECK_REFERENCE(ref) \
    NVXIO_ASSERT(ref != 0 && vxGetStatus((vx_reference)ref) == VX_SUCCESS)

/**
 * \ingroup group_nvxio_utility
 * \brief Performs a CUDA operation. If the operation fails, then it throws `std::runtime_error` exception.
 * \param [in] cudaOp Specifies a function to be called.
 * The function must have `cudaError_t` return value.
 * \see nvx_nvxio_api
 */
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

/**
 * \ingroup group_nvxio_utility
 * \brief Returns the size of an array (the \p N template argument).
 * \see nvx_nvxio_api
 */
template <typename T, vx_size N>
vx_size dimOf(T (&)[N]) { return N; }

//
// Common constants
//

/**
 * \ingroup group_nvxio_utility
 * \brief Double-precision PI.
 * \see nvx_nvxio_api
 */
const vx_float64 PI = 3.1415926535897932;
/**
 * \ingroup group_nvxio_utility
 * \brief Float-precision PI.
 * \see nvx_nvxio_api
 */
const vx_float32 PI_F = 3.14159265f;

//
// Auxiliary functions
//

#ifdef __ANDROID__
/**
 * \ingroup group_nvxio_utility
 * \brief The callback for OpenVX error logs, which prints messages to android log.
 * Must be used as a parameter for \ref vxRegisterLogCallback.
 * \param [in] context  Specifies the OpenVX context.
 * \param [in] ref      Specifies the reference to the object that generated the error message.
 * \param [in] status   Specifies the error code.
 * \param [in] string   Specifies the error message.
 */
NVXIO_EXPORT void VX_CALLBACK androidLogCallback(vx_context context, vx_reference ref, vx_status status, const vx_char string[]);
#else
/**
 * \ingroup group_nvxio_utility
 * \brief The callback for OpenVX error logs, which prints messages to standard output.
 * Must be used as a parameter for \ref vxRegisterLogCallback.
 * \param [in] context  Specifies the OpenVX context.
 * \param [in] ref      Specifies the reference to the object that generated the error message.
 * \param [in] status   Specifies the error code.
 * \param [in] string   Specifies the error message.
 */
NVXIO_EXPORT void VX_CALLBACK stdoutLogCallback(vx_context context, vx_reference ref, vx_status status, const vx_char string[]);
#endif

/**
 * \ingroup group_nvxio_utility
 * \brief Prints performance information for the provided graph object.
 * \param [in] graph    Specifies the graph object.
 * \param [in] label    Specifies the label.
 */
NVXIO_EXPORT void printPerf(vx_graph graph, const char* label);

/**
 * \ingroup group_nvxio_utility
 * \brief Prints performance information for the provided node object.
 * \param [in] node     Specifies the node object.
 * \param [in] label    Specifies the label.
 */
NVXIO_EXPORT void printPerf(vx_node node, const char* label);

/**
 * \ingroup group_nvxio_utility
 * \brief Prints Visionworks and OpenVX standard version information.
 */
NVXIO_EXPORT void printVersionInfo();

/**
 * \ingroup group_nvxio_utility
 * \brief Checks whether the context is valid and throws an exception in case of failure.
 * \param [in] context Specifies the context to check.
 * \see nvx_nvxio_api
 */
NVXIO_EXPORT void checkIfContextIsValid(vx_context context);

/**
 * \ingroup group_nvxio_utility
 * \brief `%ContextGuard` is a wrapper for `vx_context`. It is intended for safe releasing of some resources.
 * It is recommended to use `ContextGuard` in your OWR application instead of `vx_context`.
 * \see nvx_nvxio_api
 */
struct NVXIO_EXPORT ContextGuard
{
    ContextGuard() : context(vxCreateContext()) {
        checkIfContextIsValid(context);
    }
    ContextGuard(const ContextGuard &) = delete;
    ContextGuard &operator = (const ContextGuard &) = delete;
    ~ContextGuard() {
        vxReleaseContext(&context);
    }

    operator vx_context() const { return context; }
    operator vx_reference() const { return (vx_reference)context; }

private:
    vx_context context;
};

/**
 * \ingroup group_nvxio_utility
 * \brief make_unique function.
 * \see nvx_nvxio_api
 */
template <typename T, typename... Args>
std::unique_ptr<T> makeUP(Args &&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

} // namespace ovxio

#endif // NVXIO_UTILITY_HPP
