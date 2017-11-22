/*
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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

#ifndef RENDER_UTILS_HPP
#define RENDER_UTILS_HPP

#include <cmath>
#include <vector>

#include "Private/Types.hpp"

namespace nvidiaio {

inline int32_t getNumCircleSegments(float r)
{
    return static_cast<int>(10 * sqrtf(r)); //change the 10 to a smaller/bigger number as needed
}

inline void addToArray(std::vector<nvxcu_point4f_t> & array, const nvxcu_point4f_t & elem)
{
    array.push_back(elem);
}

template <typename ArrayType>
void genCircleLines(ArrayType & lines, float cx, float cy, float r,
                    int32_t num_segments, int32_t divisor = 1, float start_angle = 0.0f)
{
    float theta = 2.0f * ovxio::PI_F / num_segments;

    // precalculate the sine and cosine
    float c = cosf(theta);
    float s = sinf(theta);

    float t = 0.0f;

    // we start at angle = 0
    float x = r;
    float y = 0;

    // apply the rotation to start with start_angle
    {
        float cs = cosf(start_angle);
        float ss = sinf(start_angle);
        t = x;
        x = cs * x - ss * y;
        y = ss * t + cs * y;
    }

    num_segments = (num_segments + divisor - 1) / divisor;

    for (int32_t i = 0; i < num_segments ; i++)
    {
        // output vertex
        nvxcu_point4f_t pt;

        pt.x = x + cx;
        pt.y = y + cy;

        // apply the rotation matrix
        t = x;
        x = c * x - s * y;
        y = s * t + c * y;

        pt.z = x + cx;
        pt.w = y + cy;

        addToArray(lines, pt);
    }
}

} // namespace nvidiaio

#endif // RENDER_UTILS_HPP
