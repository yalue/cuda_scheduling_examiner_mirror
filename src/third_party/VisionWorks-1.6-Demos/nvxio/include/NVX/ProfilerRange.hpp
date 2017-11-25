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

#ifndef NVXIO_PROFILERRANGE_HPP
#define NVXIO_PROFILERRANGE_HPP

#include <stdint.h>
#include <NVX/Export.hpp>

/**
 * \file
 * \brief The `ProfilerRange` utility class.
 */

/**
 * \defgroup group_nvxio_profiler Profiler extensions
 * \ingroup nvx_nvxio_api
 */

namespace nvxio {

/**
 * \brief Push/Pop nested time ranges.
 * \ingroup group_nvxio_profiler
 * \note NVXIO must be built with NVTX support.
 */
class NVXIO_EXPORT ProfilerRange
{
public:
    /**
     * \brief Marks the start of a new time range.
     * \param [in] color        The color of new range.
     * \param [in] message      The message associated to this range event.
     */
    ProfilerRange(uint32_t color, const char* message);

    /**
     * \brief Marks the end of a time range.
     */
    ~ProfilerRange();

    ProfilerRange(const ProfilerRange&) = delete;
    ProfilerRange& operator =(const ProfilerRange&) = delete;
};

/**
 * \brief "Fuschia" color ARGB constant.
 */
const uint32_t COLOR_ARGB_FUSCHIA = 0xFFCC0066;

/**
 * \brief "Orange" color ARGB constant.
 */
const uint32_t COLOR_ARGB_ORANGE = 0xFFCC6600;

}

#endif
