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

// 0 - x
// 1 - y
// 2 - width
// 3 - height
typedef Eigen::Vector4f Rectanglef;

// 0 - x
// 1 - y
typedef Eigen::Vector2f Point2f;

static const char KERNEL_TRUNCATE_STAB_TRANSFORM_NAME[VX_MAX_KERNEL_NAME] = "example.nvx.truncate_stab_transform";

static void transformPoint(const Matrix3x3f_rm &H, const Point2f & p, Point2f & newP)
{
    float x = H(0, 0) * p(0) + H(0, 1) * p(1) + H(0, 2);
    float y = H(1, 0) * p(0) + H(1, 1) * p(1) + H(1, 2);
    float z = H(2, 0) * p(0) + H(2, 1) * p(1) + H(2, 2);

    newP(0) = x / z;
    newP(1) = y / z;
}

static bool rectContains(const Rectanglef & rect, const Point2f & pt)
{
    return rect(0) <= pt(0) && pt(0) < rect(0) + rect(2) &&
           rect(1) <= pt(1) && pt(1) < rect(1) + rect(3);
}

static bool isPointInsideCroppingRect(const Rectanglef & rect, const Matrix3x3f_rm & H, const Point2f & p)
{
    Point2f newP;
    transformPoint(H, p, newP);

    return rectContains(rect, newP);
}

static bool isMotionGood(const Matrix3x3f_rm & transform,
                         int frameWidth, int frameHeight,
                         const Matrix3x3f_rm & resizeMat, float factor)
{
    Rectanglef rect;
    rect << 0.0f, 0.0f,
            static_cast<float>(frameWidth - 1),
            static_cast<float>(frameHeight - 1);

    Matrix3x3f_rm H = (1 - factor) * transform + factor * resizeMat;

    Point2f p1, p2, p3, p4;
    p1 << 0.0f, 0.0f;
    p2 << static_cast<float>(frameWidth - 1), 0.0f;
    p3 << static_cast<float>(frameWidth - 1), static_cast<float>(frameHeight - 1);
    p4 << 0.0f, static_cast<float>(frameHeight - 1);

    return isPointInsideCroppingRect(rect, H, p1) && isPointInsideCroppingRect(rect, H, p2) &&
           isPointInsideCroppingRect(rect, H, p3) && isPointInsideCroppingRect(rect, H, p4);
}

static bool truncateTransform(const Matrix3x3f_rm & transform, int frameWidth, int frameHeight,
                              const Matrix3x3f_rm & resizeMat, Matrix3x3f_rm & truncatedTransform)
{
    float t = 0;
    if ( isMotionGood(transform, frameWidth, frameHeight, resizeMat, t) )
    {
        return false;
    }

    float l = 0, r = 1;
    while (r - l > 1e-2f)
    {
        t = (l + r) * 0.5f;
        if ( isMotionGood(transform, frameWidth, frameHeight, resizeMat, t) )
            r = t;
        else
            l = t;
    }

    truncatedTransform = (1 - t) * transform + t * resizeMat;

    return true;
}

// Kernel implementation
static vx_status VX_CALLBACK truncateStabTransform_kernel(vx_node, const vx_reference *parameters, vx_uint32 num)
{
    if (num != 4)
        return VX_FAILURE;

    vx_status status = VX_SUCCESS;

    vx_matrix vxStabTransform = (vx_matrix)parameters[0];
    vx_matrix vxTruncatedTransform = (vx_matrix)parameters[1];
    vx_image image = (vx_image)parameters[2];
    vx_scalar sCropMargin = (vx_scalar)parameters[3];

    vx_float32 stabTransformData[9] = {0};
    status |= vxCopyMatrix(vxStabTransform, stabTransformData, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    Matrix3x3f_rm stabTransform = Matrix3x3f_rm::Map(stabTransformData, 3, 3), invStabTransform;

    vx_float32 cropMargin;
    status |= vxCopyScalar(sCropMargin, &cropMargin, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    if (cropMargin < 0) // without truncation
    {
        invStabTransform = stabTransform.inverse(); // inverse the matrix for vxWarpPerspectiveNode
        status |= vxCopyMatrix(vxTruncatedTransform, invStabTransform.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
        return status;
    }

    vx_uint32 width = 0, height = 0;
    status |= vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width));
    status |= vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height));

    Matrix3x3f_rm resizeMat = Matrix3x3f_rm::Identity();
    float scale = 1.0f / (1.0f - 2 * cropMargin);
    resizeMat(0, 0) = resizeMat(1, 1) = scale;
    resizeMat(0, 2) = - scale * width * cropMargin;
    resizeMat(1, 2) = - scale * height * cropMargin;

    stabTransform.transposeInPlace(); // transpose to the standart form like resizeMat
    stabTransform = resizeMat * stabTransform;

    invStabTransform = stabTransform.inverse();
    Matrix3x3f_rm invResizeMat = resizeMat.inverse();

    Matrix3x3f_rm invTruncatedTransform;
    bool isTruncated = truncateTransform(invStabTransform, width, height, invResizeMat, invTruncatedTransform);

    if (isTruncated)
    {
        stabTransform = invTruncatedTransform.inverse();
    }

    stabTransform.transposeInPlace(); // inverse transpose
    invStabTransform = stabTransform.inverse(); // inverse the matrix for vxWarpPerspectiveNode
    status |= vxCopyMatrix(vxTruncatedTransform, invStabTransform.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

    return status;
}

// Parameter validator
static vx_status VX_CALLBACK truncateStabTransform_validate(vx_node, const vx_reference parameters[],
                                                            vx_uint32 numParams, vx_meta_format metas[])
{
    if (numParams != 4) return VX_ERROR_INVALID_PARAMETERS;

    vx_matrix stabTransform = (vx_matrix)parameters[0];
    vx_scalar cropMargin = (vx_scalar)parameters[3];

    vx_enum stabTransformDataType = 0;
    vx_size stabTransformRows = 0ul, stabTransformCols = 0ul;
    vxQueryMatrix(stabTransform, VX_MATRIX_ATTRIBUTE_TYPE, &stabTransformDataType, sizeof(stabTransformDataType));
    vxQueryMatrix(stabTransform, VX_MATRIX_ATTRIBUTE_ROWS, &stabTransformRows, sizeof(stabTransformRows));
    vxQueryMatrix(stabTransform, VX_MATRIX_ATTRIBUTE_COLUMNS, &stabTransformCols, sizeof(stabTransformCols));

    vx_enum cropMarginType = 0;
    vxQueryScalar(cropMargin, VX_SCALAR_ATTRIBUTE_TYPE, &cropMarginType, sizeof(cropMarginType));

    vx_status status = VX_SUCCESS;

    if (stabTransformDataType != VX_TYPE_FLOAT32 || stabTransformCols != 3 || stabTransformRows != 3)
    {
        status = VX_ERROR_INVALID_PARAMETERS;
    }

    if (cropMarginType == VX_TYPE_FLOAT32)
    {
        vx_float32 val = 0;
        vxCopyScalar(cropMargin, &val, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if ( val >= 0.5 )
        {
            status = VX_ERROR_INVALID_VALUE;
        }
    }
    else
    {
        status = VX_ERROR_INVALID_TYPE;
    }

    vx_meta_format truncatedTransformMeta = metas[1];

    vx_enum truncatedTransformType = VX_TYPE_FLOAT32;
    vx_size truncatedTransformRows = 3;
    vx_size truncatedTransformCols = 3;

    vxSetMetaFormatAttribute(truncatedTransformMeta, VX_MATRIX_ATTRIBUTE_TYPE, &truncatedTransformType, sizeof(truncatedTransformType));
    vxSetMetaFormatAttribute(truncatedTransformMeta, VX_MATRIX_ATTRIBUTE_ROWS, &truncatedTransformRows, sizeof(truncatedTransformRows));
    vxSetMetaFormatAttribute(truncatedTransformMeta, VX_MATRIX_ATTRIBUTE_COLUMNS, &truncatedTransformCols, sizeof(truncatedTransformCols));

    return status;
}

// Register user defined kernel in OpenVX context
vx_status registerTruncateStabTransformKernel(vx_context context)
{
    vx_status status = VX_SUCCESS;

    vx_enum id;
    status = vxAllocateUserKernelId(context, &id);
    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "[%s:%u] Failed to allocate an ID for the TruncateStabTransform kernel",
                      __FUNCTION__, __LINE__);
        return status;
    }

    vx_kernel kernel = vxAddUserKernel(context, KERNEL_TRUNCATE_STAB_TRANSFORM_NAME,
                                       id,
                                       truncateStabTransform_kernel,
                                       4,
                                       truncateStabTransform_validate,
                                       NULL,
                                       NULL
                                       );

    status = vxGetStatus((vx_reference)kernel);
    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "[%s:%u] Failed to create TruncateStabTransform Kernel", __FUNCTION__, __LINE__);
        return status;
    }

    status |= vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_MATRIX, VX_PARAMETER_STATE_REQUIRED);  // stabTransform
    status |= vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_MATRIX, VX_PARAMETER_STATE_REQUIRED); // truncatedTransform
    status |= vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);   // image
    status |= vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED);  // cropMargin

    if (status != VX_SUCCESS)
    {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status, "[%s:%u] Failed to initialize TruncateStabTransform Kernel parameters", __FUNCTION__, __LINE__);
        return VX_FAILURE;
    }

    status = vxFinalizeKernel(kernel);
    vxReleaseKernel(&kernel);

    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "[%s:%u] Failed to finalize TruncateStabTransform Kernel", __FUNCTION__, __LINE__);
        return VX_FAILURE;
    }

    return status;
}

vx_node truncateStabTransformNode(vx_graph graph, vx_matrix stabTransform, vx_matrix truncatedTransform, vx_image image, vx_scalar cropMargin)
{
    vx_node node = NULL;

    vx_kernel kernel = vxGetKernelByName(vxGetContext((vx_reference)graph), KERNEL_TRUNCATE_STAB_TRANSFORM_NAME);

    if (vxGetStatus((vx_reference)kernel) == VX_SUCCESS)
    {
        node = vxCreateGenericNode(graph, kernel);
        vxReleaseKernel(&kernel);

        if (vxGetStatus((vx_reference)node) == VX_SUCCESS)
        {
            vxSetParameterByIndex(node, 0, (vx_reference)stabTransform);
            vxSetParameterByIndex(node, 1, (vx_reference)truncatedTransform);
            vxSetParameterByIndex(node, 2, (vx_reference)image);
            vxSetParameterByIndex(node, 3, (vx_reference)cropMargin);
        }
    }

    return node;
}
