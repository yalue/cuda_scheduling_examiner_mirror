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

#include "vstab_nodes.hpp"

#include <Eigen/SVD>

static const char KERNEL_HOMOGRAPHY_FILTER_NAME[VX_MAX_KERNEL_NAME] = "example.nvx.homography_filter";

// Kernel implementation
static vx_status VX_CALLBACK homographyFilter_kernel(vx_node, const vx_reference *parameters, vx_uint32 num)
{
    if (num != 4)
        return VX_FAILURE;

    vx_status status = VX_SUCCESS;

    vx_matrix input = (vx_matrix)parameters[0];
    vx_matrix homography = (vx_matrix)parameters[1];
    vx_image image = (vx_image)parameters[2];
    vx_array mask = (vx_array)parameters[3];

    // Copy input to homography
    vx_float32 intputData[9] = {0};
    status |= vxCopyMatrix(input, intputData, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyMatrix(homography, intputData, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

    vx_uint32 width = 0, height = 0;
    status |= vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width));
    status |= vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height));

    vx_size nPoints;
    status |= vxQueryArray(mask, VX_ARRAY_ATTRIBUTE_NUMITEMS, &nPoints, sizeof(nPoints));

    vx_int32 nInliers = 0;
    if (nPoints > 0)
    {
        vx_map_id map_id;
        vx_size stride;
        void* ptr;
        status |= vxMapArrayRange(mask, 0, nPoints, &map_id, &stride, &ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);

        for (vx_size i = 0; i < nPoints; i++)
        {
            vx_uint8 v = vxArrayItem(vx_uint8, ptr, i, stride);
            if (v != 0)
                ++nInliers;
        }

        status |= vxUnmapArrayRange(mask, map_id);
    }

    int inlierThresh = std::max(15, static_cast<int>(0.1 * nPoints));
    Matrix3x3f_rm eye3x3 = Matrix3x3f_rm::Identity();

    if (nInliers < inlierThresh)
    {
        status |= vxCopyMatrix(homography, eye3x3.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
        return status;
    }

    vx_float32 data[9];
    status |= vxCopyMatrix(homography, data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    Matrix3x3f_rm M = Matrix3x3f_rm::Map(data, 3, 3);
    M.transposeInPlace();

    // restrictions on the lenghts of the diagonals of the warped image
    Matrix3x4f_rm vertices = Matrix3x4f_rm::Zero();

    for(int i=0; i<4; ++i)
        vertices(2, i) = 1.0f;

    vertices(0, 1) = static_cast<float>(width);
    vertices(0, 2) = static_cast<float>(width);
    vertices(1, 2) = static_cast<float>(height);
    vertices(1, 3) = static_cast<float>(height);

    Matrix3x4f_rm dstVertices = M * vertices;
    for(int i=0; i<4; ++i)
    {
        dstVertices(0,i) /= dstVertices(2,i);
        dstVertices(1,i) /= dstVertices(2,i);
        dstVertices(2,i) = 1.0f;
    }

    float diagLenGold = std::sqrt(static_cast<float>(width*width + height*height));

    float dx = dstVertices(0,0) - dstVertices(0,2);
    float dy = dstVertices(1,0) - dstVertices(1,2);
    float lenDiag1 = sqrt(dx*dx + dy*dy);

    dx = dstVertices(0,1) - dstVertices(0,3);
    dy = dstVertices(1,1) - dstVertices(1,3);
    float lenDiag2 = sqrt(dx*dx + dy*dy);

    float averDiagLen = (lenDiag1 + lenDiag2) / 2;
    float diagRatio1 = std::min(diagLenGold, averDiagLen) / std::max(diagLenGold, averDiagLen);
    if (diagRatio1 < 0.5f)
    {
        status |= vxCopyMatrix(homography, eye3x3.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
        return status;
    }

    float maxDiag = std::max(lenDiag1, lenDiag2);
    if (maxDiag > 0.0f)
    {
        float diagRatio2 = std::min(lenDiag1, lenDiag2) / maxDiag;
        if (diagRatio2 < 0.25f)
        {
            status |= vxCopyMatrix(homography, eye3x3.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
            return status;
        }
    }
    else
    {
        status |= vxCopyMatrix(homography, eye3x3.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
        return status;
    }

    // restriction on min eigen value
    typedef Eigen::JacobiSVD<Matrix3x3f_rm> JacobiSVD;

    JacobiSVD svd(M);
    JacobiSVD::SingularValuesType singValues = svd.singularValues();

    if (singValues(2) < 1e-4f)
    {
        status |= vxCopyMatrix(homography, eye3x3.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
        return status;
    }

    return status;
}

// Parameter validator
static vx_status VX_CALLBACK homographyFilter_validate(vx_node, const vx_reference parameters[],
                                                       vx_uint32 numParams, vx_meta_format metas[])
{
    if (numParams != 4) return VX_ERROR_INVALID_PARAMETERS;

    vx_matrix input = (vx_matrix)parameters[0];
    vx_array mask = (vx_array)parameters[3];

    vx_status status = VX_SUCCESS;

    vx_enum inputDataType = 0;
    vx_size inputRows = 0ul, inputCols = 0ul;
    vxQueryMatrix(input, VX_MATRIX_ATTRIBUTE_TYPE, &inputDataType, sizeof(inputDataType));
    vxQueryMatrix(input, VX_MATRIX_ATTRIBUTE_ROWS, &inputRows, sizeof(inputRows));
    vxQueryMatrix(input, VX_MATRIX_ATTRIBUTE_COLUMNS, &inputCols, sizeof(inputCols));

    vx_enum maskType = 0;
    vxQueryArray(mask, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &maskType, sizeof(maskType));

    if (inputDataType != VX_TYPE_FLOAT32 || inputCols != 3 || inputRows != 3)
    {
        status = VX_ERROR_INVALID_PARAMETERS;
    }

    if (maskType != VX_TYPE_UINT8)
    {
        status = VX_ERROR_INVALID_TYPE;
    }

    vx_meta_format homographyMeta = metas[1];

    vx_enum homographyType = VX_TYPE_FLOAT32;
    vx_size homographyRows = 3;
    vx_size homographyCols = 3;

    vxSetMetaFormatAttribute(homographyMeta, VX_MATRIX_ATTRIBUTE_TYPE, &homographyType, sizeof(homographyType));
    vxSetMetaFormatAttribute(homographyMeta, VX_MATRIX_ATTRIBUTE_ROWS, &homographyRows, sizeof(homographyRows));
    vxSetMetaFormatAttribute(homographyMeta, VX_MATRIX_ATTRIBUTE_COLUMNS, &homographyCols, sizeof(homographyCols));

    return status;
}

// Register user defined kernel in OpenVX context
vx_status registerHomographyFilterKernel(vx_context context)
{
    vx_status status = VX_SUCCESS;

    vx_enum id;
    status = vxAllocateUserKernelId(context, &id);
    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "[%s:%u] Failed to allocate an ID for the HomographyFilter kernel",
                      __FUNCTION__, __LINE__);
        return status;
    }

    vx_kernel kernel = vxAddUserKernel(context, KERNEL_HOMOGRAPHY_FILTER_NAME,
                                       id,
                                       homographyFilter_kernel,
                                       4,
                                       homographyFilter_validate,
                                       NULL,
                                       NULL
                                       );

    status = vxGetStatus((vx_reference)kernel);
    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "[%s:%u] Failed to create HomographyFilter Kernel", __FUNCTION__, __LINE__);
        return status;
    }

    status |= vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_MATRIX, VX_PARAMETER_STATE_REQUIRED); // input
    status |= vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_MATRIX, VX_PARAMETER_STATE_REQUIRED); // homography
    status |= vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED); // image
    status |= vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED); // mask

    if (status != VX_SUCCESS)
    {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status, "[%s:%u] Failed to initialize HomographyFilter Kernel parameters", __FUNCTION__, __LINE__);
        return VX_FAILURE;
    }

    status = vxFinalizeKernel(kernel);
    vxReleaseKernel(&kernel);

    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "[%s:%u] Failed to finalize HomographyFilter Kernel", __FUNCTION__, __LINE__);
        return VX_FAILURE;
    }

    return status;
}


vx_node homographyFilterNode(vx_graph graph, vx_matrix input, vx_matrix homography, vx_image image, vx_array mask)
{
    vx_node node = NULL;

    vx_kernel kernel = vxGetKernelByName(vxGetContext((vx_reference)graph), KERNEL_HOMOGRAPHY_FILTER_NAME);

    if (vxGetStatus((vx_reference)kernel) == VX_SUCCESS)
    {
        node = vxCreateGenericNode(graph, kernel);
        vxReleaseKernel(&kernel);

        if (vxGetStatus((vx_reference)node) == VX_SUCCESS)
        {
            vxSetParameterByIndex(node, 0, (vx_reference)input);
            vxSetParameterByIndex(node, 1, (vx_reference)homography);
            vxSetParameterByIndex(node, 2, (vx_reference)image);
            vxSetParameterByIndex(node, 3, (vx_reference)mask);
        }
    }

    return node;
}
