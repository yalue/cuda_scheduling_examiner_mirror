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

#include "stabilizer.hpp"

#include <climits>
#include <cfloat>
#include <iostream>
#include <iomanip>

#include <VX/vxu.h>
#include <NVX/nvx.h>

#include <OVX/UtilityOVX.hpp>

#include "vstab_nodes.hpp"

namespace
{
    class ImageBasedVideoStabilizer : public nvx::VideoStabilizer
    {
    public:
        ImageBasedVideoStabilizer(vx_context context, const VideoStabilizerParams& params);
        ~ImageBasedVideoStabilizer();

        void init(vx_image firstFrame);
        void process(vx_image newFrame);

        vx_image getStabilizedFrame() const;

        void printPerfs() const;

    private:

        struct HarrisPyrLKParams
        {
            vx_size pyr_levels;

            vx_float32 harris_k;
            vx_float32 harris_thresh;
            vx_uint32 harris_cell_size;

            vx_uint32 lk_num_iters;
            vx_size lk_win_size;

            HarrisPyrLKParams();
        };

        void processFirstFrame(vx_image frame);
        void createMainGraph(vx_image frame);

        void createDataObjects(vx_image frame);
        void release();

        VideoStabilizerParams vstabParams_;
        HarrisPyrLKParams harrisParams_;

        vx_graph graph_;
        vx_context context_;

        // Format for current frames
        vx_df_image format_;
        vx_uint32 width_;
        vx_uint32 height_;

        // Node from main graph (used to print performance results)
        vx_node convert_to_gray_node_;
        vx_node copy_node_;
        vx_node pyr_node_;
        vx_node opt_flow_node_;
        vx_node feature_track_node_;
        vx_node find_homography_node_;
        vx_node homography_filter_node_;
        vx_node matrix_smoother_node_;
        vx_node truncate_stab_transform_node_;
        vx_node warp_perspective_node_;

        vx_delay pyr_delay_;
        vx_delay pts_delay_;
        vx_delay matrices_delay_;
        vx_delay frames_RGBX_delay_;

        vx_matrix smoothed_;

        vx_image stabilized_RGBX_frame_;

        vx_scalar s_lk_epsilon_;
        vx_scalar s_lk_num_iters_;
        vx_scalar s_lk_use_init_est_;
        vx_scalar s_crop_margin_;

        vx_size matrices_delay_size_;
        vx_size frames_delay_size_;
    };

    ImageBasedVideoStabilizer::ImageBasedVideoStabilizer(vx_context context, const VideoStabilizerParams &params):
        vstabParams_(params)
    {
        context_ = context;
        graph_ = 0;

        format_ = VX_DF_IMAGE_VIRT;
        width_ = 0;
        height_ = 0;

        convert_to_gray_node_ = 0;
        copy_node_ = 0;
        pyr_node_ = 0;
        opt_flow_node_ = 0;
        feature_track_node_ = 0;
        find_homography_node_ = 0;
        homography_filter_node_ = 0;
        matrix_smoother_node_ = 0;
        truncate_stab_transform_node_ = 0;
        warp_perspective_node_ = 0;

        pyr_delay_ = 0;
        pts_delay_ = 0;
        matrices_delay_ = 0;
        frames_RGBX_delay_ = 0;

        smoothed_ = 0;
        stabilized_RGBX_frame_ = 0;

        s_lk_epsilon_ = 0;
        s_lk_num_iters_ = 0;
        s_lk_use_init_est_ = 0;
        s_crop_margin_ = 0;

        matrices_delay_size_ = 0;
        frames_delay_size_ = 0;
    }

    void ImageBasedVideoStabilizer::init(vx_image firstFrame)
    {
        vx_df_image format = VX_DF_IMAGE_VIRT;
        vx_uint32 width = 0;
        vx_uint32 height = 0;

        NVXIO_SAFE_CALL( vxQueryImage(firstFrame, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)) );
        NVXIO_SAFE_CALL( vxQueryImage(firstFrame, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)) );
        NVXIO_SAFE_CALL( vxQueryImage(firstFrame, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)) );

        NVXIO_ASSERT(format == VX_DF_IMAGE_RGBX);

        release();

        format_ = format;
        width_ = width;
        height_ = height;

        createDataObjects(firstFrame);
        createMainGraph(firstFrame);

        processFirstFrame(firstFrame);
    }

    void ImageBasedVideoStabilizer::processFirstFrame(vx_image frame)
    {
        vx_image gray = vxCreateImage(context_, width_, height_, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(gray);

        NVXIO_SAFE_CALL( vxuColorConvert(context_, frame, gray) );
        NVXIO_SAFE_CALL( nvxuCopyImage(context_, frame, (vx_image)vxGetReferenceFromDelay(frames_RGBX_delay_, 0)) );

        NVXIO_SAFE_CALL( vxuGaussianPyramid(context_, gray,
                                        (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, 0)) );
        NVXIO_SAFE_CALL( nvxuHarrisTrack(context_, gray,
                                         (vx_array)vxGetReferenceFromDelay(pts_delay_, 0), NULL, 0,
                                         harrisParams_.harris_k, harrisParams_.harris_thresh, harrisParams_.harris_cell_size, NULL) );

        vxReleaseImage(&gray);
    }

    void ImageBasedVideoStabilizer::process(vx_image newFrame)
    {
        // Check input format
        vx_df_image format = VX_DF_IMAGE_VIRT;
        vx_uint32 width = 0;
        vx_uint32 height = 0;

        NVXIO_SAFE_CALL( vxQueryImage(newFrame, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)) );
        NVXIO_SAFE_CALL( vxQueryImage(newFrame, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)) );
        NVXIO_SAFE_CALL( vxQueryImage(newFrame, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)) );

        NVXIO_ASSERT(format == format_);
        NVXIO_ASSERT(width == width_);
        NVXIO_ASSERT(height == height_);

        // Update frame queue
        NVXIO_SAFE_CALL( vxAgeDelay(pyr_delay_) );
        NVXIO_SAFE_CALL( vxAgeDelay(pts_delay_) );
        NVXIO_SAFE_CALL( vxAgeDelay(matrices_delay_) );
        NVXIO_SAFE_CALL( vxAgeDelay(frames_RGBX_delay_) );

        // Process graph
        NVXIO_SAFE_CALL( vxSetParameterByIndex(convert_to_gray_node_, 0, (vx_reference)newFrame) );
        NVXIO_SAFE_CALL( vxSetParameterByIndex(copy_node_, 0, (vx_reference)newFrame) );

        NVXIO_SAFE_CALL( vxProcessGraph(graph_) );
    }

    void ImageBasedVideoStabilizer::createMainGraph(vx_image frame)
    {
        NVXIO_SAFE_CALL( registerMatrixSmootherKernel(context_) );
        NVXIO_SAFE_CALL( registerHomographyFilterKernel(context_) );
        NVXIO_SAFE_CALL( registerTruncateStabTransformKernel(context_) );

        graph_ = vxCreateGraph(context_);
        NVXIO_CHECK_REFERENCE(graph_);

        vx_image gray = vxCreateVirtualImage(graph_, 0, 0, VX_DF_IMAGE_U8);
        NVXIO_CHECK_REFERENCE(gray);

        //vxColorConvertNode
        convert_to_gray_node_ = vxColorConvertNode(graph_, frame, gray);
        NVXIO_CHECK_REFERENCE(convert_to_gray_node_);

        //nvxCopyImageNode
        copy_node_ = nvxCopyImageNode(graph_, frame, (vx_image)vxGetReferenceFromDelay(frames_RGBX_delay_, 0));
        NVXIO_CHECK_REFERENCE(copy_node_);

        //vxGaussianPyramidNode
        pyr_node_ = vxGaussianPyramidNode(graph_, gray,
                                          (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, 0));
        NVXIO_CHECK_REFERENCE(pyr_node_);

        vx_array kp_curr_list = vxCreateVirtualArray(graph_, NVX_TYPE_POINT2F, 1000);
        NVXIO_CHECK_REFERENCE(kp_curr_list);

        //vxOpticalFlowPyrLKNode
        opt_flow_node_ = vxOpticalFlowPyrLKNode(graph_,
            (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, -1), (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, 0),
            (vx_array)vxGetReferenceFromDelay(pts_delay_, -1), (vx_array)vxGetReferenceFromDelay(pts_delay_, -1),
            kp_curr_list, VX_TERM_CRITERIA_BOTH, s_lk_epsilon_, s_lk_num_iters_, s_lk_use_init_est_, harrisParams_.lk_win_size);
        NVXIO_CHECK_REFERENCE(opt_flow_node_);

        //nvxFindHomographyNode
        vx_matrix homography = vxCreateMatrix(context_, VX_TYPE_FLOAT32, 3, 3);
        vx_array mask = vxCreateVirtualArray(graph_, VX_TYPE_UINT8, 1000);
        find_homography_node_ = nvxFindHomographyNode(graph_, (vx_array)vxGetReferenceFromDelay(pts_delay_, -1),
                                                      kp_curr_list,
                                                      homography,
                                                      NVX_FIND_HOMOGRAPHY_METHOD_RANSAC, 3.0f,
                                                      2000, 10,
                                                      0.995f, 0.45f,
                                                      mask);
        NVXIO_CHECK_REFERENCE(find_homography_node_);

        //homographyFilterNode

        homography_filter_node_ = homographyFilterNode(graph_, homography,
                                                       (vx_matrix)vxGetReferenceFromDelay(matrices_delay_, 0),
                                                       frame, mask);
        NVXIO_CHECK_REFERENCE(homography_filter_node_);

        //matrixSmootherNode
        matrix_smoother_node_ = matrixSmootherNode(graph_, matrices_delay_, smoothed_);
        NVXIO_CHECK_REFERENCE(matrix_smoother_node_);

        //truncateStabTransformNode
        vx_matrix truncated = vxCreateMatrix(context_, VX_TYPE_FLOAT32, 3, 3);
        truncate_stab_transform_node_ = truncateStabTransformNode(graph_, smoothed_, truncated, frame, s_crop_margin_);
        NVXIO_CHECK_REFERENCE(truncate_stab_transform_node_);

        //vxWarpPerspectiveNode
        warp_perspective_node_ = vxWarpPerspectiveNode(graph_,
                (vx_image)vxGetReferenceFromDelay(frames_RGBX_delay_, 1 - static_cast<vx_int32>(frames_delay_size_)),
                truncated,
                VX_INTERPOLATION_TYPE_BILINEAR, stabilized_RGBX_frame_);
        NVXIO_CHECK_REFERENCE(warp_perspective_node_);

        //nvxHarrisTrackNode
        feature_track_node_ = nvxHarrisTrackNode(graph_, gray,
                                                 (vx_array)vxGetReferenceFromDelay(pts_delay_, 0), NULL,
                                                 kp_curr_list, harrisParams_.harris_k, harrisParams_.harris_thresh, harrisParams_.harris_cell_size, NULL);
        NVXIO_CHECK_REFERENCE(feature_track_node_);

        // Ensure highest graph optimization level
        const char* option = "-O3";
        NVXIO_SAFE_CALL( vxSetGraphAttribute(graph_, NVX_GRAPH_VERIFY_OPTIONS, option, strlen(option)) );

        NVXIO_SAFE_CALL( vxVerifyGraph(graph_) );

        vxReleaseMatrix(&homography);
        vxReleaseMatrix(&truncated);

        vxReleaseArray(&kp_curr_list);
        vxReleaseArray(&mask);
        vxReleaseImage(&gray);
    }
}

void ImageBasedVideoStabilizer::printPerfs() const
{
    vx_perf_t perf;

    NVXIO_SAFE_CALL( vxQueryGraph(graph_, VX_GRAPH_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
    std::cout << "Graph Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

    NVXIO_SAFE_CALL( vxQueryNode(convert_to_gray_node_, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
    std::cout << "\t RGB to gray time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

    NVXIO_SAFE_CALL( vxQueryNode(copy_node_, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
    std::cout << "\t Copy time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

    NVXIO_SAFE_CALL( vxQueryNode(pyr_node_, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
    std::cout << "\t Pyramid time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

    NVXIO_SAFE_CALL( vxQueryNode(opt_flow_node_, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
    std::cout << "\t Optical Flow time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

    NVXIO_SAFE_CALL( vxQueryNode(find_homography_node_, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
    std::cout << "\t Find Homography time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

    NVXIO_SAFE_CALL( vxQueryNode(homography_filter_node_, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
    std::cout << "\t Homography Filter time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

    NVXIO_SAFE_CALL( vxQueryNode(matrix_smoother_node_, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
    std::cout << "\t Matrices Smoothing time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

    NVXIO_SAFE_CALL( vxQueryNode(truncate_stab_transform_node_, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
    std::cout << "\t Truncate Stab Transform time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

    NVXIO_SAFE_CALL( vxQueryNode(warp_perspective_node_, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
    std::cout << "\t Warp Perspective time: " << perf.tmp / 1000000.0 << " ms" << std::endl;

    NVXIO_SAFE_CALL( vxQueryNode(feature_track_node_, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
    std::cout << "\t Feature Track time : " << perf.tmp / 1000000.0 << " ms" << std::endl;
}

static vx_status initDelayOfMatrices(vx_delay delayOfMatrices)
{
    vx_status status = VX_SUCCESS;

    vx_enum type = 0;
    status |= vxQueryDelay(delayOfMatrices, VX_DELAY_ATTRIBUTE_TYPE, &type, sizeof(type));
    NVXIO_ASSERT(type == VX_TYPE_MATRIX);

    vx_size size = 0;
    status |= vxQueryDelay(delayOfMatrices, VX_DELAY_ATTRIBUTE_SLOTS, &size, sizeof(size));

    vx_float32 eye[9] = {1,0,0, 0,1,0, 0,0,1};
    for (vx_int32 i = 1 - static_cast<vx_int32>(size); i <= 0 && status == VX_SUCCESS; ++i)
    {
        vx_matrix mat = (vx_matrix)vxGetReferenceFromDelay(delayOfMatrices, i);
        status |= vxCopyMatrix(mat, eye, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    }
    return status;
}

vx_status nvx::initDelayOfImages(vx_context context, vx_delay delayOfImages)
{
    vx_status status = VX_SUCCESS;

    vx_enum type = 0;
    status |= vxQueryDelay(delayOfImages, VX_DELAY_ATTRIBUTE_TYPE, &type, sizeof(type));
    NVXIO_ASSERT(type == VX_TYPE_IMAGE);

    vx_size size = 0;
    status |= vxQueryDelay(delayOfImages, VX_DELAY_ATTRIBUTE_SLOTS, &size, sizeof(size));
    if (size > 0)
    {
        vx_df_image format = VX_DF_IMAGE_VIRT;
        vx_uint32 width = 0, height = 0;

        vx_image img0 = (vx_image)vxGetReferenceFromDelay(delayOfImages, 0);
        status |= vxQueryImage(img0, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format));
        status |= vxQueryImage(img0, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width));
        status |= vxQueryImage(img0, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height));

        NVXIO_ASSERT(format == VX_DF_IMAGE_RGBX);

        vx_pixel_value_t initVal;
        initVal.RGBX[0] = 0;
        initVal.RGBX[1] = 0;
        initVal.RGBX[2] = 0;
        initVal.RGBX[3] = 0;
        vx_image blackImg = vxCreateUniformImage(context, width, height, format, &initVal);
        NVXIO_CHECK_REFERENCE(blackImg);

        for (vx_int32 i = 1 - static_cast<vx_int32>(size); i < 0 && status == VX_SUCCESS; ++i)
        {
            vx_image img = (vx_image)vxGetReferenceFromDelay(delayOfImages, i);
            status |= nvxuCopyImage(context, blackImg, img);
        }
        vxReleaseImage(&blackImg);
    }

    return status;
}

void ImageBasedVideoStabilizer::createDataObjects(vx_image frame)
{
    vx_pyramid pyr_exemplar = vxCreatePyramid(context_, harrisParams_.pyr_levels, VX_SCALE_PYRAMID_HALF, width_, height_, VX_DF_IMAGE_U8);
    NVXIO_CHECK_REFERENCE(pyr_exemplar);
    vx_array pts_exemplar = vxCreateArray(context_, NVX_TYPE_POINT2F, 1000);
    NVXIO_CHECK_REFERENCE(pts_exemplar);

    pyr_delay_ = vxCreateDelay(context_, (vx_reference)pyr_exemplar, 2);
    NVXIO_CHECK_REFERENCE(pyr_delay_);

    pts_delay_ = vxCreateDelay(context_, (vx_reference)pts_exemplar, 2);
    NVXIO_CHECK_REFERENCE(pts_delay_);

    vxReleasePyramid(&pyr_exemplar);
    vxReleaseArray(&pts_exemplar);

    smoothed_ = vxCreateMatrix(context_, VX_TYPE_FLOAT32, 3, 3);
    NVXIO_CHECK_REFERENCE(smoothed_);

    matrices_delay_size_ = 2 * vstabParams_.numOfSmoothingFrames_ + 1;
    matrices_delay_ = vxCreateDelay(context_, (vx_reference)smoothed_, matrices_delay_size_);
    NVXIO_CHECK_REFERENCE(matrices_delay_);
    NVXIO_SAFE_CALL( initDelayOfMatrices(matrices_delay_) );

    vx_image image_exemplar = vxCreateImage(context_, width_, height_, VX_DF_IMAGE_U8);

    // 'frames_delay_' must have such size to be synchronized with the 'matrices_delay_'
    frames_delay_size_ = vstabParams_.numOfSmoothingFrames_ + 2;

    frames_RGBX_delay_ = vxCreateDelay(context_, (vx_reference)frame, frames_delay_size_);
    NVXIO_CHECK_REFERENCE(frames_RGBX_delay_);
    NVXIO_SAFE_CALL( nvx::initDelayOfImages(context_, frames_RGBX_delay_) );

    vxReleaseImage(&image_exemplar);

    stabilized_RGBX_frame_ = vxCreateImage(context_, width_, height_, VX_DF_IMAGE_RGBX);
    NVXIO_CHECK_REFERENCE(stabilized_RGBX_frame_);

    vx_float32 lk_epsilon = 0.01f;
    s_lk_epsilon_ = vxCreateScalar(context_, VX_TYPE_FLOAT32, &lk_epsilon);
    NVXIO_CHECK_REFERENCE(s_lk_epsilon_);

    s_lk_num_iters_ = vxCreateScalar(context_, VX_TYPE_UINT32, &harrisParams_.lk_num_iters);
    NVXIO_CHECK_REFERENCE(s_lk_num_iters_);

    vx_bool lk_use_init_est = vx_false_e;
    s_lk_use_init_est_ = vxCreateScalar(context_, VX_TYPE_BOOL, &lk_use_init_est);
    NVXIO_CHECK_REFERENCE(s_lk_use_init_est_);

    s_crop_margin_ = vxCreateScalar(context_, VX_TYPE_FLOAT32, &vstabParams_.cropMargin_);
    NVXIO_CHECK_REFERENCE(s_crop_margin_);
}

void ImageBasedVideoStabilizer::release()
{
    format_ = VX_DF_IMAGE_VIRT;
    width_ = 0;
    height_ = 0;

    vxReleaseDelay(&pyr_delay_);
    vxReleaseDelay(&pts_delay_);

    vxReleaseNode(&pyr_node_);
    vxReleaseNode(&opt_flow_node_);
    vxReleaseNode(&feature_track_node_);
    vxReleaseNode(&find_homography_node_);
    vxReleaseNode(&homography_filter_node_);
    vxReleaseNode(&matrix_smoother_node_);
    vxReleaseNode(&truncate_stab_transform_node_);
    vxReleaseNode(&warp_perspective_node_);

    vxReleaseDelay(&matrices_delay_);
    vxReleaseDelay(&frames_RGBX_delay_);
    vxReleaseMatrix(&smoothed_);

    vxReleaseNode(&convert_to_gray_node_);
    vxReleaseNode(&copy_node_);

    vxReleaseImage(&stabilized_RGBX_frame_);
    vxReleaseScalar(&s_lk_epsilon_);
    vxReleaseScalar(&s_lk_num_iters_);
    vxReleaseScalar(&s_lk_use_init_est_);
    vxReleaseScalar(&s_crop_margin_);

    vxReleaseGraph(&graph_);
}

nvx::VideoStabilizer::VideoStabilizerParams::VideoStabilizerParams()
{
    numOfSmoothingFrames_ = 5;
    cropMargin_ = 0.05f;
}

ImageBasedVideoStabilizer::HarrisPyrLKParams::HarrisPyrLKParams()
{
    pyr_levels = 6;

    harris_k = 0.04f;
    harris_thresh = 100.0f;
    harris_cell_size = 18;

    lk_num_iters = 5;
    lk_win_size = 10;
}

nvx::VideoStabilizer* nvx::VideoStabilizer::createImageBasedVStab(vx_context context, const VideoStabilizerParams &params)
{
    return new ImageBasedVideoStabilizer(context, params);
}

vx_image ImageBasedVideoStabilizer::getStabilizedFrame() const
{
    return stabilized_RGBX_frame_;
}

ImageBasedVideoStabilizer::~ImageBasedVideoStabilizer()
{
    release();
}
