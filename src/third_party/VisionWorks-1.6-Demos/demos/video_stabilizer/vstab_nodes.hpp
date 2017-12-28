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

#ifndef NVX_VSTAB_NODES_HPP
#define NVX_VSTAB_NODES_HPP

#include <NVX/nvx.h>

#include <algorithm>
#include <Eigen/Dense>

// row-major storage order
typedef Eigen::Matrix<vx_float32, 3, 3, Eigen::RowMajor> Matrix3x3f_rm;
typedef Eigen::Matrix<vx_float32, 3, 4, Eigen::RowMajor> Matrix3x4f_rm;

// Register homographyFilter kernel in OpenVX context
vx_status registerHomographyFilterKernel(vx_context context);

// Create homographyFilter node
vx_node homographyFilterNode(vx_graph graph, vx_matrix input,
                             vx_matrix homography, vx_image image,
                             vx_array mask);


// Register matrixSmoother kernel in OpenVX context
vx_status registerMatrixSmootherKernel(vx_context context);

// Create matrixSmoother node
vx_node matrixSmootherNode(vx_graph graph,
                      vx_delay matrices, vx_matrix smoothed);


// Register truncateStabTransform kernel in OpenVX context
vx_status registerTruncateStabTransformKernel(vx_context context);

/* Create truncateStabTransform node.
 * cropMargin - proportion of the width(height) of the frame
 * that is allowed to be cropped for stabilizing of the frames. The value should be less than 0.5.
 * If cropMargin is negative then the truncation procedure is turned off.
 */
vx_node truncateStabTransformNode(vx_graph graph, vx_matrix stabTransform, vx_matrix truncatedTransform,
                                  vx_image image, vx_scalar cropMargin);

#endif
