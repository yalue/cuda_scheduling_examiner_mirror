/*
# Copyright (c) 2014-2015, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NVX_STABILIZER_HPP
#define NVX_STABILIZER_HPP

#include <VX/vx.h>

namespace nvx
{
    class VideoStabilizer
    {
    public:

        struct VideoStabilizerParams
        {
            // frames for smoothing are taken from the interval [-numOfSmoothingFrames_; numOfSmoothingFrames_] in the current frame's vicinity
            vx_size numOfSmoothingFrames_;
            // proportion of the width/height of the frame that is allowed to be cropped for stabilizing of the frames
            vx_float32 cropMargin_;

            VideoStabilizerParams();
        };

        static VideoStabilizer* createImageBasedVStab(vx_context context, const VideoStabilizerParams& params = VideoStabilizerParams());

        virtual ~VideoStabilizer() {}

        virtual void init(vx_image firstFrame) = 0;
        virtual void process(vx_image newFrame) = 0;

        virtual vx_image getStabilizedFrame() const = 0;

        virtual void printPerfs() const = 0;
    };

    vx_status initDelayOfImages(vx_context context, vx_delay delayOfImages);
}

#endif
