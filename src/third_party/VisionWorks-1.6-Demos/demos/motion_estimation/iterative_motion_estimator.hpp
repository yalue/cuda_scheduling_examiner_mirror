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

#ifndef ITERATIVE_MOTION_ESTIMATOR_HPP
#define ITERATIVE_MOTION_ESTIMATOR_HPP

#include <vector>
#include <VX/vx.h>

class IterativeMotionEstimator
{
public:
    struct Params
    {
        vx_float32 biasWeight;
        vx_int32 mvDivFactor;
        vx_float32 smoothnessFactor;

	// target device
	vx_enum targetDevice;

        Params();
    };

    explicit IterativeMotionEstimator(vx_context context);
    ~IterativeMotionEstimator();

    void init(vx_image prevFrameRGBX, vx_image currFrameRGBX, const Params& params = Params());
    void release();

    void process();

    vx_image getMotionField() const;

    void printPerfs() const;

private:
    void createDataObjects(vx_image prevFrameRGBX, vx_image currFrameRGBX);
    void processInitialFrame();
    void createMainGraph();

private:
    vx_context context_;

    Params params_;

    // Format for current frames
    vx_df_image format_;
    vx_uint32 width_;
    vx_uint32 height_;

    // Resulting motion field
    vx_image mfOut_;

    // Input/output ROIs (with dimensions aligned to 32)
    vx_uint32 widthROI_;
    vx_uint32 heightROI_;
    vx_image prevFrameROI_;
    vx_image currFrameROI_;
    vx_image mfOutROI_;

    // Two successive pyramids
    vx_delay pyr_delay_;

    // Main graph
    vx_graph graph_;

    // Nodes
    vx_node cvt_color_node_;
    vx_node pyramid_node_;
    std::vector<vx_node> create_mf_nodes_;
    std::vector<vx_node> refine_mf_nodes_;
    std::vector<vx_node> partition_mf_nodes_;
    std::vector<vx_node> mult_mf_nodes_;
};


#endif
