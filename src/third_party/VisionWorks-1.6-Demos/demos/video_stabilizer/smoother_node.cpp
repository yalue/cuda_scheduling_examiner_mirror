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

#include <vector>

static const char KERNEL_MATRIX_SMOOTHER_NAME[VX_MAX_KERNEL_NAME] = "example.nvx.matrix_smoother";

//
// Define user kernel
//

static Matrix3x3f_rm getTransformation(const std::vector<Matrix3x3f_rm>& mats, vx_int32 from, vx_int32 to)
{
    Matrix3x3f_rm M = Matrix3x3f_rm::Identity();

    if (to > from)
    {
        for (vx_int32 i = from; i < to; ++i)
            M = M * mats[i];
    }
    else if (to < from)
    {
        for (vx_int32 i = to; i < from; ++i)
            M = M * mats[i];

        M = Matrix3x3f_rm(M.inverse());
    }

    return M;
}

static void getCompensatingTransformation(const std::vector<vx_matrix>& transforms, vx_int32 idx,
                                          vx_int32 smoothingWindow, vx_matrix transform)
{
    vx_int32 num = static_cast<vx_int32>(transforms.size());

    std::vector<vx_float32> gaussWeights(num);
    vx_float32 sigma = smoothingWindow * 0.7f;
    for(vx_int32 i = -smoothingWindow; i < num-smoothingWindow; ++i)
    {
        gaussWeights[i+smoothingWindow] = exp( - i * i / (2.f * sigma * sigma) );
    }

    vx_float32 sum = 0.0f;
    for (vx_int32 i = 0; i < num; ++i)
        sum += gaussWeights[i];

    for (vx_int32 i = 0; i < num; ++i)
        gaussWeights[i] /= sum;

    std::vector<Matrix3x3f_rm> mats;
    mats.reserve(num);

    vx_float32 data[9];
    for (vx_int32 i = 0; i < num; ++i)
    {
        vxCopyMatrix(transforms[i], data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        mats.push_back( Matrix3x3f_rm::Map(data, 3, 3) );
    }

    Matrix3x3f_rm avg = Matrix3x3f_rm::Zero();
    for (vx_int32 i = idx-smoothingWindow; i <= idx+smoothingWindow; ++i)
        avg += gaussWeights[i - idx + smoothingWindow] * getTransformation(mats, idx, i);

    vxCopyMatrix(transform, avg.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
}

// Kernel implementation
static vx_status VX_CALLBACK matrixSmoother_kernel(vx_node, const vx_reference *parameters, vx_uint32)
{
    vx_delay delay = (vx_delay)parameters[0];
    vx_size numInputParams;
    vxQueryDelay(delay, VX_DELAY_ATTRIBUTE_SLOTS, &numInputParams, sizeof(numInputParams));

    std::vector<vx_matrix> matrices(numInputParams);
    for(vx_size i=0; i<numInputParams; ++i)
    {
        matrices[i] = (vx_matrix)vxGetReferenceFromDelay(delay, i + 1 - (vx_int32)numInputParams);
    }

    vx_matrix output = (vx_matrix)parameters[1];

    vx_int32 smoothingWindow = numInputParams / 2;
    vx_int32 idx = smoothingWindow;

    getCompensatingTransformation(matrices, idx, smoothingWindow, output);

    return VX_SUCCESS;
}

// Parameter validator
static vx_status VX_CALLBACK matrixSmoother_validate(vx_node, const vx_reference parameters[],
                                                     vx_uint32 numParams, vx_meta_format metas[])
{
    if (numParams != 2) return VX_ERROR_INVALID_PARAMETERS;

    vx_delay matrices = (vx_delay)parameters[0];

    vx_enum matricesType = VX_TYPE_INVALID;
    vxQueryDelay(matrices, VX_DELAY_ATTRIBUTE_TYPE, &matricesType, sizeof(matricesType));

    vx_status status = VX_SUCCESS;

    if (matricesType == VX_TYPE_MATRIX)
    {
        vx_matrix matrix = (vx_matrix)vxGetReferenceFromDelay(matrices, 0);
        vx_enum type = 0;
        vxQueryMatrix(matrix, VX_MATRIX_ATTRIBUTE_TYPE, &type, sizeof(type));
        vx_size cols = 0, rows = 0;
        vxQueryMatrix(matrix, VX_MATRIX_ATTRIBUTE_COLUMNS, &cols, sizeof(cols));
        vxQueryMatrix(matrix, VX_MATRIX_ATTRIBUTE_ROWS, &rows, sizeof(rows));

        if (type != VX_TYPE_FLOAT32 || cols != 3 || rows != 3)
        {
            status = VX_ERROR_INVALID_PARAMETERS;
        }
    }
    else
    {
        status = VX_ERROR_INVALID_TYPE;
    }

    vx_meta_format smoothedMeta = metas[1];

    vx_enum smoothedType = VX_TYPE_FLOAT32;
    vx_size smoothedCols = 3, smoothedRows = 3;

    vxSetMetaFormatAttribute(smoothedMeta, VX_MATRIX_ATTRIBUTE_TYPE, &smoothedType, sizeof(smoothedType) );
    vxSetMetaFormatAttribute(smoothedMeta, VX_MATRIX_ATTRIBUTE_ROWS, &smoothedRows, sizeof(smoothedRows) );
    vxSetMetaFormatAttribute(smoothedMeta, VX_MATRIX_ATTRIBUTE_COLUMNS, &smoothedCols, sizeof(smoothedCols) );

    return status;
}

// Register user defined kernel in OpenVX context
vx_status registerMatrixSmootherKernel(vx_context context)
{
    vx_status status = VX_SUCCESS;

    vx_enum id;
    status = vxAllocateUserKernelId(context, &id);
    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "[%s:%u] Failed to allocate an ID for the MatrixSmoother kernel",
                      __FUNCTION__, __LINE__);
        return status;
    }

    vx_kernel kernel = vxAddUserKernel(context, KERNEL_MATRIX_SMOOTHER_NAME,
                                       id,
                                       matrixSmoother_kernel,
                                       2,
                                       matrixSmoother_validate,
                                       NULL,
                                       NULL
                                       );

    status = vxGetStatus((vx_reference)kernel);
    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "[%s:%u] Failed to create MatrixSmoother Kernel", __FUNCTION__, __LINE__);
        return status;
    }

    status |= vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_DELAY, VX_PARAMETER_STATE_REQUIRED);
    status |= vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_MATRIX, VX_PARAMETER_STATE_REQUIRED);

    if (status != VX_SUCCESS)
    {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status, "[%s:%u] Failed to initialize MatrixSmoother Kernel parameters", __FUNCTION__, __LINE__);
        return VX_FAILURE;
    }

    status = vxFinalizeKernel(kernel);
    vxReleaseKernel(&kernel);

    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "[%s:%u] Failed to finalize MatrixSmoother Kernel", __FUNCTION__, __LINE__);
        return VX_FAILURE;
    }

    return status;
}


vx_node matrixSmootherNode(vx_graph graph, vx_delay matrices, vx_matrix smoothed)
{
    vx_node node = NULL;

    vx_kernel kernel = vxGetKernelByName(vxGetContext((vx_reference)graph), KERNEL_MATRIX_SMOOTHER_NAME);

    if (vxGetStatus((vx_reference)kernel) == VX_SUCCESS)
    {
        node = vxCreateGenericNode(graph, kernel);
        vxReleaseKernel(&kernel);

        if (vxGetStatus((vx_reference)node) == VX_SUCCESS)
        {
            vxSetParameterByIndex(node, 0, (vx_reference)matrices);
            vxSetParameterByIndex(node, 1, (vx_reference)smoothed);
        }
    }

    return node;
}
