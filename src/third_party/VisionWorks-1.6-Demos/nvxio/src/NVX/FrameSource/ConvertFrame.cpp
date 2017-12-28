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

#include "Private/Types.hpp"

#include <NVX/ProfilerRange.hpp>

#include <cuda_runtime_api.h>
#include "FrameSource/FrameSourceImpl.hpp"
#include <gst/app/gstappsink.h>
#include <sstream>
#include <iostream>

namespace nvidiaio
{

void convertFrame(nvxcu_stream_exec_target_t &exec_target,
                  const image_t & image,
                  const FrameSource::Parameters & configuration,
                  int width, int height,
                  bool usePitch, size_t pitch,
                  int depth, void * decodedPtr,
                  bool is_cuda,
                  void *& devMem,
                  size_t & devMemPitch)

{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "ConvertFrame (NVXIO)");

    cudaStream_t stream = nullptr;
    bool needConvert = image.format != configuration.format;
    bool canCopyDirectly = !needConvert ||
            (configuration.format == NVXCU_DF_IMAGE_NV12 && image.format == NVXCU_DF_IMAGE_U8);

    // allocate CUDA memory to copy decoded image to
    if (!is_cuda && !canCopyDirectly)
    {
        if (!devMem)
        {
            size_t height_dec = height;

            if (configuration.format == NVXCU_DF_IMAGE_NV12)
                height_dec += height_dec >> 1;

            // we assume that decoded image will have no more than 4 channels per pixel
            NVXIO_ASSERT( cudaSuccess == cudaMallocPitch(&devMem, &devMemPitch,
                                                         width * 4, height_dec) );
        }
    }

    void * devMems[2] = { devMem, nullptr };

    if (!canCopyDirectly)
    {
        NVXIO_ASSERT(needConvert);

        if (!is_cuda)
        {
            NVXIO_ASSERT(devMem);

            // a. upload decoded image to CUDA buffer
            if (configuration.format == NVXCU_DF_IMAGE_U8 ||
                    configuration.format == NVXCU_DF_IMAGE_RGB ||
                    configuration.format == NVXCU_DF_IMAGE_RGBX)
            {
                vx_int32 stride_y = usePitch ? pitch : ((width * depth + 3) >> 2) << 2;
                vx_int32 stride_x = depth;

                CUDA_SAFE_CALL (
                        cudaMemcpy2DAsync(devMems[0], devMemPitch,
                                          decodedPtr, stride_y,
                                          width * stride_x,
                                          height, cudaMemcpyHostToDevice, stream) );
                CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );
            }

            else if (configuration.format == NVXCU_DF_IMAGE_NV12)
            {
                vx_int32 stride_y = usePitch ? pitch : ((width + 3) >> 2) << 2;
                vx_int32 stride_x = sizeof(uint8_t);
                CUDA_SAFE_CALL (
                        cudaMemcpy2DAsync(devMems[0], devMemPitch,
                                          decodedPtr, stride_y,
                                          width * stride_x,
                                          height, cudaMemcpyHostToDevice, stream) );


                devMems[1] = (uint8_t *)devMem + devMemPitch * height;
                stride_x = sizeof(uint16_t);
                decodedPtr = (void *) ((uint8_t *)decodedPtr + stride_y * height);
                width >>= 1;
                height >>= 1;
                CUDA_SAFE_CALL (
                            cudaMemcpy2DAsync(devMems[1], devMemPitch,
                                              decodedPtr, stride_y,
                                              width * stride_x,
                                              height, cudaMemcpyHostToDevice, stream) );
                CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );
            }
            else
            {
                NVXIO_THROW_EXCEPTION("Unsupported image format");
            }
        }
        else
        {
            if (configuration.format == NVXCU_DF_IMAGE_U8 ||
                    configuration.format == NVXCU_DF_IMAGE_RGB ||
                    configuration.format == NVXCU_DF_IMAGE_RGBX)
            {
                vx_int32 stride_y = usePitch ? pitch : ((width * depth + 3) >> 2) << 2;
                devMems[0] = decodedPtr;
                devMemPitch = stride_y;
            }

            else if (configuration.format == NVXCU_DF_IMAGE_NV12)
            {
                vx_int32 stride_y = usePitch ? pitch : ((width + 3) >> 2) << 2;
                devMems[0] = decodedPtr;
                devMemPitch = stride_y;
                devMems[1] = (void *) ((uint8_t *)decodedPtr + stride_y * height);


            }
            else
            {
                NVXIO_THROW_EXCEPTION("Unsupported image format");
            }
        }
    }



    if (canCopyDirectly)
    {


        cudaMemcpyKind copyKind = is_cuda ? cudaMemcpyDeviceToDevice :
                                            cudaMemcpyHostToDevice;

        void * framePtr = nullptr;

        if (configuration.format == NVXCU_DF_IMAGE_U8 ||
                configuration.format == NVXCU_DF_IMAGE_RGB ||
                configuration.format == NVXCU_DF_IMAGE_RGBX)
        {
            vx_int32 stride_y = usePitch ? pitch : ((width * depth + 3) >> 2) << 2;
            vx_int32 stride_x = depth;
            framePtr = image.planes[0].ptr;
            CUDA_SAFE_CALL (
                        cudaMemcpy2DAsync(framePtr, image.planes[0].pitch_in_bytes,
                        decodedPtr, stride_y,
                        width * stride_x,
                        height, copyKind, stream) );

            CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );
        }
        else if (configuration.format == NVXCU_DF_IMAGE_NV12 && image.format == NVXCU_DF_IMAGE_NV12)
        {
            vx_int32 stride_y = usePitch ? pitch : ((width + 3) >> 2) << 2;
            vx_int32 stride_x = sizeof(uint8_t);;
            framePtr = image.planes[0].ptr;
            CUDA_SAFE_CALL (
                        cudaMemcpy2DAsync(framePtr, image.planes[0].pitch_in_bytes,
                        decodedPtr, stride_y,
                        width * stride_x,
                        height, copyKind, stream) );

            framePtr = image.planes[1].ptr;
            stride_x = sizeof(uint16_t);
            width >>= 1;
            height >>= 1;
            decodedPtr = (void *) ((uint8_t *)decodedPtr + stride_y * height);
            CUDA_SAFE_CALL (
                        cudaMemcpy2DAsync(framePtr, image.planes[1].pitch_in_bytes,
                        decodedPtr, stride_y,
                        width * stride_x,
                        height, copyKind, stream) );
            CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );
        }
        else
        {
            NVXIO_THROW_EXCEPTION("Unsupported image format");
        }
    }
    else
    {
        nvxcu_pitch_linear_image_t input, output;
        input.base.format = configuration.format;
        input.base.width = configuration.frameWidth;
        input.base.height = configuration.frameHeight;
        input.base.image_type = NVXCU_PITCH_LINEAR_IMAGE;
        NVXIO_ASSERT(input.base.format == NVXCU_DF_IMAGE_U8 ||
                     input.base.format == NVXCU_DF_IMAGE_RGB ||
                     input.base.format == NVXCU_DF_IMAGE_RGBX ||
                     input.base.format == NVXCU_DF_IMAGE_NV12);
        input.planes[0].dev_ptr = devMems[0];
        input.planes[0].pitch_in_bytes = devMemPitch;
        if (input.base.format == NVXCU_DF_IMAGE_NV12)
        {
            input.planes[1].dev_ptr = devMems[1];
            input.planes[1].pitch_in_bytes = devMemPitch;
        }

        output.base.format = image.format;
        output.base.width = image.width;
        output.base.height = image.height;
        output.base.image_type = NVXCU_PITCH_LINEAR_IMAGE;
        NVXIO_ASSERT(output.base.format == NVXCU_DF_IMAGE_U8 ||
                     output.base.format == NVXCU_DF_IMAGE_RGB ||
                     output.base.format == NVXCU_DF_IMAGE_RGBX ||
                     output.base.format == NVXCU_DF_IMAGE_NV12);
        output.planes[0].dev_ptr = image.planes[0].ptr;
        output.planes[0].pitch_in_bytes = image.planes[0].pitch_in_bytes;
        if (output.base.format == NVXCU_DF_IMAGE_NV12)
        {
            output.planes[1].dev_ptr = image.planes[1].ptr;
            output.planes[1].pitch_in_bytes = image.planes[1].pitch_in_bytes;
        }


        NVXCU_SAFE_CALL( nvxcuColorConvert(&input.base, &output.base, NVXCU_COLOR_SPACE_DEFAULT,
                                           NVXCU_CHANNEL_RANGE_FULL, &exec_target.base) );


    }


}

} // namespace nvidiaio
