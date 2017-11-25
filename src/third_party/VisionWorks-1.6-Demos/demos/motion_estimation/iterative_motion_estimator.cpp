/*
# Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
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

#include "iterative_motion_estimator.hpp"

#include <iostream>
#include <iomanip>
#include <vector>

#include <NVX/nvx.h>

#include "OVX/UtilityOVX.hpp"

struct Size
{
    vx_int32 width, height;
};

const vx_int32 NUM_LEVELS = 3;

const Size winSizePerLevel[NUM_LEVELS] = {
    {16, 16},
    {32, 16},
    {48, 32}
};

const vx_int32 numItersPerLevel[NUM_LEVELS] = {
    6,
    4,
    4
};

IterativeMotionEstimator::Params::Params()
{
    biasWeight = 1.0f;
    mvDivFactor = 4;
    smoothnessFactor = 1.0f;
}

IterativeMotionEstimator::IterativeMotionEstimator(vx_context context)
{
    context_ = context;
    NVXIO_SAFE_CALL( vxRetainReference((vx_reference)context_) );

    format_ = VX_DF_IMAGE_VIRT;
    width_ = 0;
    height_ = 0;

    mfOut_ = nullptr;

    widthROI_ = 0;
    heightROI_ = 0;
    prevFrameROI_ = nullptr;
    currFrameROI_ = nullptr;
    mfOutROI_ = nullptr;

    pyr_delay_ = nullptr;

    graph_ = nullptr;

    cvt_color_node_ = nullptr;
    pyramid_node_ = nullptr;
}

IterativeMotionEstimator::~IterativeMotionEstimator()
{
    release();

    vxReleaseContext(&context_);
}

void IterativeMotionEstimator::init(vx_image prevFrameRGBX, vx_image currFrameRGBX, const Params& params)
{
    // Check input format

    vx_df_image format = VX_DF_IMAGE_VIRT;
    vx_uint32 width = 0;
    vx_uint32 height = 0;

    NVXIO_SAFE_CALL( vxQueryImage(prevFrameRGBX, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)) );
    NVXIO_SAFE_CALL( vxQueryImage(prevFrameRGBX, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)) );
    NVXIO_SAFE_CALL( vxQueryImage(prevFrameRGBX, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)) );

    NVXIO_ASSERT(format == VX_DF_IMAGE_RGBX);
    NVXIO_ASSERT(width >= 32 && height >= 32);

    // Re-create graph

    release();

    params_ = params;

    format_ = format;
    width_ = width;
    height_ = height;

    createDataObjects(prevFrameRGBX, currFrameRGBX);
    createMainGraph();
    processInitialFrame();
}

void IterativeMotionEstimator::release()
{
    format_ = VX_DF_IMAGE_VIRT;
    width_ = 0;
    height_ = 0;

    vxReleaseImage(&mfOut_);

    vxReleaseImage(&prevFrameROI_);
    vxReleaseImage(&currFrameROI_);
    vxReleaseImage(&mfOutROI_);

    vxReleaseDelay(&pyr_delay_);

    vxReleaseGraph(&graph_);
}

// This function creates data objects that are not entirely linked to graphs
void IterativeMotionEstimator::createDataObjects(vx_image prevFrameRGBX, vx_image currFrameRGBX)
{
    // Resulting motion field.
    // The algorithm calculates motion field for 2x2 pixel blocks,
    // that's why the motion field object is created with scale factor 2.

    vx_uint32 mf_width = (width_ + 1) / 2;
    vx_uint32 mf_height = (height_ + 1) / 2;

    mfOut_ = vxCreateImage(context_, mf_width, mf_height, NVX_DF_IMAGE_2F32);
    NVXIO_CHECK_REFERENCE(mfOut_);

    // Fill the motion field with zeros

    {
        vx_imagepatch_addressing_t addr;
        addr.dim_x = mf_width;
        addr.dim_y = mf_height;
        addr.stride_x = 2*sizeof(vx_float32);
        addr.stride_y = addr.stride_x*addr.dim_x;
        std::vector<vx_float32> buf(mf_width*mf_height*2, 0.0f);
        NVXIO_SAFE_CALL( vxCopyImagePatch(mfOut_, NULL, 0, &addr, buf.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST) );
    }

    // Create ROIs with dimensions aligned to 32

    widthROI_ = (width_ / 32) * 32;
    heightROI_ = (height_ / 32) * 32;

    vx_rectangle_t frame_rect = {
        0u, 0u,
        widthROI_, heightROI_
    };

    prevFrameROI_ = vxCreateImageFromROI(prevFrameRGBX, &frame_rect);
    NVXIO_CHECK_REFERENCE(prevFrameROI_);

    currFrameROI_ = vxCreateImageFromROI(currFrameRGBX, &frame_rect);
    NVXIO_CHECK_REFERENCE(currFrameROI_);

    vx_rectangle_t mf_rect = {
        0u, 0u,
        vx_uint32(widthROI_ / 2),
        vx_uint32(heightROI_ / 2)
    };

    mfOutROI_ = vxCreateImageFromROI(mfOut_, &mf_rect);
    NVXIO_CHECK_REFERENCE(mfOutROI_);

    // Two successive pyramids are necessary for the computation.
    // A delay object with 2 slots is created for this purpose.

    vx_pyramid pyr_exemplar = vxCreatePyramid(context_, NUM_LEVELS, VX_SCALE_PYRAMID_HALF, widthROI_, heightROI_, VX_DF_IMAGE_U8);
    NVXIO_CHECK_REFERENCE(pyr_exemplar);

    pyr_delay_ = vxCreateDelay(context_, (vx_reference)pyr_exemplar, 2);
    NVXIO_CHECK_REFERENCE(pyr_delay_);

    vxReleasePyramid(&pyr_exemplar);
}

//
// See motion_estimation_user_guide.md for explanation
//
void IterativeMotionEstimator::createMainGraph()
{
    graph_ = nvxCreateStreamGraph(context_);
    NVXIO_CHECK_REFERENCE(graph_);

    vx_pyramid prev_pyr = (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, -1);
    vx_pyramid curr_pyr = (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, 0);

    // Color Convert

    vx_image frame_gray = vxCreateVirtualImage(graph_, 0, 0, VX_DF_IMAGE_U8);
    NVXIO_CHECK_REFERENCE(frame_gray);

    cvt_color_node_ = vxColorConvertNode(graph_, currFrameROI_, frame_gray);
    NVXIO_CHECK_REFERENCE(cvt_color_node_);

    // Gaussian Pyramid

    pyramid_node_ = vxGaussianPyramidNode(graph_, frame_gray, curr_pyr);
    NVXIO_CHECK_REFERENCE(pyramid_node_);

    vxReleaseImage(&frame_gray);

    // Virtual buffers

    //
    // To optimize memory usage by the pipeline we will reuse virtual buffers across all levels of pyramid.
    // For that purpose we use stream graph, since it allows a data object to be written multiple times.
    //
    // We create 1 buffer for SAD table and 4 buffers for motion fields with sizes for largest level.
    // All intermediate data objects will be created as sub ROIs of that buffers in the following way:
    //
    // prev_lvl_mf (mf_bufs[2]) -> [CreateMotionField] -> mf_0 (mf_bufs[0])
    //                                                    mf_1 (mf_bufs[1])
    //                                                    sad_table (sad_table_buf)
    //
    // mf_0 (mf_bufs[0])         -> [RefineMotionField] -> mf_refine_0 (mf_bufs[2])
    // mf_1 (mf_bufs[1])                                   mf_refine_1 (mf_bufs[3])
    // sad_table (sad_table_buf)
    //
    // mf_refine_0 (mf_bufs[2]) -> [PartitionMotionField 4x4] -> mf_partition_0 (mf_bufs[0])
    // mf_refine_1 (mf_bufs[3])                                  mf_partition_1 (mf_bufs[1])
    //
    // mf_partition_0 (mf_bufs[0]) -> [MultiplyByScalar] -> prev_lvl_mf (mf_bufs[2])
    //
    // mf_partition_0 (mf_bufs[0]) -> [PartitionMotionField 2x2] -> prev_lvl_mf (mf_bufs[3])
    // mf_partition_1 (mf_bufs[1])
    //

    vx_image sad_table_buf = vxCreateVirtualImage(graph_,
                                                  (widthROI_ / 8) * winSizePerLevel[NUM_LEVELS - 1].width * winSizePerLevel[NUM_LEVELS - 1].height,
                                                  heightROI_ / 8,
                                                  VX_DF_IMAGE_U32);
    NVXIO_CHECK_REFERENCE(sad_table_buf);

    vx_image mf_bufs[4];

    mf_bufs[0] = vxCreateVirtualImage(graph_, widthROI_ / 4, heightROI_ / 4, NVX_DF_IMAGE_2S16);
    NVXIO_CHECK_REFERENCE(mf_bufs[0]);

    mf_bufs[1] = vxCreateVirtualImage(graph_, widthROI_ / 4, heightROI_ / 4, NVX_DF_IMAGE_2S16);
    NVXIO_CHECK_REFERENCE(mf_bufs[1]);

    mf_bufs[2] = vxCreateVirtualImage(graph_, widthROI_ / 4, heightROI_ / 4, NVX_DF_IMAGE_2S16);
    NVXIO_CHECK_REFERENCE(mf_bufs[2]);

    mf_bufs[3] = vxCreateVirtualImage(graph_, widthROI_ / 2, heightROI_ / 2, NVX_DF_IMAGE_2S16);
    NVXIO_CHECK_REFERENCE(mf_bufs[3]);

    // Loop over levels

    create_mf_nodes_.resize(NUM_LEVELS);
    refine_mf_nodes_.resize(NUM_LEVELS);
    partition_mf_nodes_.resize(NUM_LEVELS + 1);
    mult_mf_nodes_.resize(NUM_LEVELS);

    vx_image prev_lvl_mf = nullptr;

    for (vx_int32 level = NUM_LEVELS - 1; level >= 0; --level)
    {
        vx_image prev_frame = vxGetPyramidLevel(prev_pyr, level);
        vx_image curr_frame = vxGetPyramidLevel(curr_pyr, level);

        vx_uint32 cur_lvl_width = 0u, cur_level_height = 0u;
        NVXIO_SAFE_CALL( vxQueryImage(prev_frame, VX_IMAGE_ATTRIBUTE_WIDTH, &cur_lvl_width, sizeof(cur_lvl_width)) );
        NVXIO_SAFE_CALL( vxQueryImage(prev_frame, VX_IMAGE_ATTRIBUTE_HEIGHT, &cur_level_height, sizeof(cur_level_height)) );

        Size winSize = winSizePerLevel[level];
        vx_int32 numIters = numItersPerLevel[level];

        // Create Motion Field

        vx_rectangle_t mf_8x8_roi = {
            0u, 0u,
            vx_uint32(cur_lvl_width / 8),
            vx_uint32(cur_level_height / 8)
        };

        vx_image mf_0 = vxCreateImageFromROI(mf_bufs[0], &mf_8x8_roi);
        NVXIO_CHECK_REFERENCE(mf_0);

        vx_image mf_1 = vxCreateImageFromROI(mf_bufs[1], &mf_8x8_roi);
        NVXIO_CHECK_REFERENCE(mf_1);

        vx_rectangle_t sad_table_roi = {
            0u, 0u,
            vx_uint32((cur_lvl_width / 8) * winSize.width * winSize.height),
            vx_uint32(cur_level_height / 8)
        };

        vx_image sad_table = vxCreateImageFromROI(sad_table_buf, &sad_table_roi);
        NVXIO_CHECK_REFERENCE(sad_table);

        create_mf_nodes_[level] = nvxCreateMotionFieldNode(graph_,
                                                           prev_frame, curr_frame,
                                                           prev_lvl_mf /*anchor*/, prev_lvl_mf /*bias*/,
                                                           mf_0, mf_1,
                                                           sad_table,
                                                           8 /*blockSize*/,
                                                           winSize.width, winSize.height,
                                                           params_.biasWeight,
                                                           params_.mvDivFactor);
        NVXIO_CHECK_REFERENCE(create_mf_nodes_[level]);

        vxReleaseImage(&prev_lvl_mf);

        // Refine Motion Field

        vx_image mf_refine_0 = vxCreateImageFromROI(mf_bufs[2], &mf_8x8_roi);
        NVXIO_CHECK_REFERENCE(mf_refine_0);

        vx_image mf_refine_1 = vxCreateImageFromROI(mf_bufs[3], &mf_8x8_roi);
        NVXIO_CHECK_REFERENCE(mf_refine_1);

        refine_mf_nodes_[level] = nvxRefineMotionFieldNode(graph_,
                                                           mf_0, mf_1,
                                                           sad_table,
                                                           mf_refine_0, mf_refine_1,
                                                           winSize.width, winSize.height,
                                                           numIters,
                                                           params_.smoothnessFactor,
                                                           params_.mvDivFactor);
        NVXIO_CHECK_REFERENCE(refine_mf_nodes_[level]);

        vxReleaseImage(&mf_0);
        vxReleaseImage(&mf_1);
        vxReleaseImage(&sad_table);

        // Partition Motion Field

        vx_rectangle_t mf_4x4_roi = {
            0u, 0u,
            vx_uint32(cur_lvl_width / 4),
            vx_uint32(cur_level_height / 4)
        };

        vx_image mf_partition_0 = vxCreateImageFromROI(mf_bufs[0], &mf_4x4_roi);
        NVXIO_CHECK_REFERENCE(mf_partition_0);

        vx_image mf_partition_1 = vxCreateImageFromROI(mf_bufs[1], &mf_4x4_roi);
        NVXIO_CHECK_REFERENCE(mf_partition_1);

        partition_mf_nodes_[level + 1] = nvxPartitionMotionFieldNode(graph_,
                                                                     prev_frame, curr_frame,
                                                                     mf_refine_0, mf_refine_1,
                                                                     mf_partition_0, mf_partition_1,
                                                                     params_.smoothnessFactor,
                                                                     params_.mvDivFactor);
        NVXIO_CHECK_REFERENCE(partition_mf_nodes_[level + 1]);

        vxReleaseImage(&mf_refine_0);
        vxReleaseImage(&mf_refine_1);

        if (level > 0)
        {
            // Upscale motion field

            prev_lvl_mf = vxCreateImageFromROI(mf_bufs[2], &mf_4x4_roi);
            NVXIO_CHECK_REFERENCE(prev_lvl_mf);

            mult_mf_nodes_[level] = nvxMultiplyByScalarNode(graph_, mf_partition_0, prev_lvl_mf, 2.0f);
            NVXIO_CHECK_REFERENCE(mult_mf_nodes_[level]);

            vxReleaseImage(&mf_partition_0);
            vxReleaseImage(&mf_partition_1);
        }
        else
        {
            // Partition Motion Field

            vx_rectangle_t mf_2x2_roi = {
                0u, 0u,
                vx_uint32(cur_lvl_width / 2),
                vx_uint32(cur_level_height / 2)
            };

            prev_lvl_mf = vxCreateImageFromROI(mf_bufs[3], &mf_2x2_roi);
            NVXIO_CHECK_REFERENCE(prev_lvl_mf);

            partition_mf_nodes_[level] = nvxPartitionMotionFieldNode(graph_,
                                                                     prev_frame, curr_frame,
                                                                     mf_partition_0, mf_partition_1,
                                                                     prev_lvl_mf, nullptr,
                                                                     params_.smoothnessFactor,
                                                                     params_.mvDivFactor);

            NVXIO_CHECK_REFERENCE(partition_mf_nodes_[level]);

            vxReleaseImage(&mf_partition_0);
            vxReleaseImage(&mf_partition_1);
        }

        vxReleaseImage(&prev_frame);
        vxReleaseImage(&curr_frame);
    }

    vxReleaseImage(&sad_table_buf);

    vxReleaseImage(&mf_bufs[0]);
    vxReleaseImage(&mf_bufs[1]);
    vxReleaseImage(&mf_bufs[2]);
    vxReleaseImage(&mf_bufs[3]);

    // Convert final motion field

    mult_mf_nodes_[0] = nvxMultiplyByScalarNode(graph_, prev_lvl_mf, mfOutROI_, 0.25f);
    NVXIO_CHECK_REFERENCE(mult_mf_nodes_[0]);

    vxReleaseImage(&prev_lvl_mf);

    // Ensure highest graph optimization level
    const char* option = "-O3";
    NVXIO_SAFE_CALL( vxSetGraphAttribute(graph_, NVX_GRAPH_VERIFY_OPTIONS, option, strlen(option)) );

    // Verify the graph

    NVXIO_SAFE_CALL( vxRegisterAutoAging(graph_, pyr_delay_) );

    NVXIO_SAFE_CALL( vxVerifyGraph(graph_) );
}

void IterativeMotionEstimator::processInitialFrame()
{
    vx_pyramid prev_pyr = (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, -1);

    vx_image frame_gray = vxCreateImage(context_, widthROI_, heightROI_, VX_DF_IMAGE_U8);
    NVXIO_CHECK_REFERENCE(frame_gray);

    NVXIO_SAFE_CALL( vxuColorConvert(context_, prevFrameROI_, frame_gray) );
    NVXIO_SAFE_CALL( vxuGaussianPyramid(context_, frame_gray, prev_pyr) );

    vxReleaseImage(&frame_gray);
}

void IterativeMotionEstimator::process()
{
    // Process graph
    NVXIO_SAFE_CALL( vxProcessGraph(graph_) );
}

vx_image IterativeMotionEstimator::getMotionField() const
{
    return mfOut_;
}

void IterativeMotionEstimator::printPerfs() const
{
    vx_perf_t perf;

    NVXIO_SAFE_CALL( vxQueryGraph(graph_, VX_GRAPH_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
    std::cout << "Graph Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

    NVXIO_SAFE_CALL( vxQueryNode(cvt_color_node_, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
    std::cout << "\t Color Convert Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

    NVXIO_SAFE_CALL( vxQueryNode(pyramid_node_, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
    std::cout << "\t Pyramid Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

    vx_uint64 create_mf_total = 0;
    vx_uint64 refine_mf_total = 0;
    vx_uint64 partition_mf_total = 0;
    vx_uint64 scale_mf_total = 0;

    for (vx_int32 i = 0; i < NUM_LEVELS; ++i)
    {
        NVXIO_SAFE_CALL( vxQueryNode(create_mf_nodes_[i], VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
        create_mf_total += perf.tmp;

        NVXIO_SAFE_CALL( vxQueryNode(refine_mf_nodes_[i], VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
        refine_mf_total += perf.tmp;

        NVXIO_SAFE_CALL( vxQueryNode(partition_mf_nodes_[i], VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
        partition_mf_total += perf.tmp;

        NVXIO_SAFE_CALL( vxQueryNode(mult_mf_nodes_[i], VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
        scale_mf_total += perf.tmp;
    }

    NVXIO_SAFE_CALL( vxQueryNode(partition_mf_nodes_[NUM_LEVELS], VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
    partition_mf_total += perf.tmp;

    std::cout << "\t Create Motion Field (x " << NUM_LEVELS << ") Time : " << create_mf_total / 1000000.0 << " ms" << std::endl;
    std::cout << "\t Refine Motion Field (x " << NUM_LEVELS << ") Time : " << refine_mf_total / 1000000.0 << " ms" << std::endl;
    std::cout << "\t Partition Motion Field (x " << NUM_LEVELS + 1 << ") Time : " << partition_mf_total / 1000000.0 << " ms" << std::endl;
    std::cout << "\t Scale Motion Field (x " << NUM_LEVELS << ") Time : " << scale_mf_total / 1000000.0 << " ms" << std::endl;
}
