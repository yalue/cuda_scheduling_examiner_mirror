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

#include <OVX/UtilityOVX.hpp>

namespace ovxio
{

#ifdef __ANDROID__
void VX_CALLBACK androidLogCallback(vx_context /*context*/, vx_reference /*ref*/, vx_status /*status*/, const vx_char string[])
{
    NVXIO_LOGE("NVX", "[NVX LOG] %s", string);
}
#else
void VX_CALLBACK stdoutLogCallback(vx_context /*context*/, vx_reference /*ref*/, vx_status /*status*/, const vx_char string[])
{
    std::cout << "[NVX LOG] " << string << std::endl;
}
#endif

void printPerf(vx_graph graph, const char* label)
{
    vx_perf_t perf;
    NVXIO_SAFE_CALL( vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );

#ifdef __ANDROID__
    NVXIO_LOGI("PERF", "%s Graph Time : %f ms", label, perf.tmp / 1000000.0);
#else
    std::cout << label << " Graph Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;
#endif
}

void printPerf(vx_node node, const char* label)
{
    vx_perf_t perf;
    NVXIO_SAFE_CALL( vxQueryNode(node, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );

#ifdef __ANDROID__
    NVXIO_LOGI("PERF", "\t %s Time : %f ms", label, perf.tmp / 1000000.0);
#else
    std::cout << "\t " << label << " Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;
#endif
}

void printVersionInfo()
{
    nvx_version_info_t info;
    nvxGetVersionInfo(&info);

    std::cout << "VisionWorks library info:" << std::endl;
    std::cout << "\t VisionWorks version : "
              << info.visionworks_version.major << "."
              << info.visionworks_version.minor << "."
              << info.visionworks_version.patch
              << info.visionworks_version.suffix << std::endl;
    std::cout << "\t OpenVX Standard version : "
              << info.openvx_major_version << "."
              << info.openvx_minor_version << "."
              << info.openvx_patch_version << std::endl;
    std::cout << std::endl;
}

void checkIfContextIsValid(vx_context context)
{
    vx_status status = vxGetStatus((vx_reference)context);

    if (status == NVX_ERROR_NO_CUDA_GPU)
    {
        NVXIO_THROW_EXCEPTION("CUDA-capable GPU was not found.");
    }
    if (status == NVX_ERROR_UNSUPPORTED_CUDA_GPU)
    {
        NVXIO_THROW_EXCEPTION("Unsupported GPU. Only Kepler and newer generation is supported for now.");
    }

    NVXIO_CHECK_REFERENCE(context);
}

}
