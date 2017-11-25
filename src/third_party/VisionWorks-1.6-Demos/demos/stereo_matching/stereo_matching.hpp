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

#ifndef __NVX_STEREO_HPP__
#define __NVX_STEREO_HPP__

#include <VX/vx.h>

class StereoMatching
{
public:

    enum ImplementationType
    {
        HIGH_LEVEL_API,
        LOW_LEVEL_API,
        LOW_LEVEL_API_PYRAMIDAL
    };

    struct StereoMatchingParams
    {
        // disparity range
        vx_int32 min_disparity;
        vx_int32 max_disparity;

        // discontinuity penalties
        vx_int32 P1;
        vx_int32 P2;

        // SAD window size
        vx_int32 sad;

        // Census Transform window size
        vx_int32 ct_win_size;

        // Hamming cost window size
        vx_int32 hc_win_size;

        // BT-cost clip value
        vx_int32 bt_clip_value;

        // validation threshold
        vx_int32 max_diff; // cross-check
        vx_int32 uniqueness_ratio;

        vx_enum scanlines_mask;

        vx_enum flags;

        StereoMatchingParams();
    };

    static StereoMatching* createStereoMatching(vx_context context, const StereoMatchingParams& params,
                                                ImplementationType impl,
                                                vx_image left, vx_image right, vx_image disparity);

    virtual ~StereoMatching() {}

    virtual void run() = 0;

    virtual void printPerfs() const = 0;
};

#endif
