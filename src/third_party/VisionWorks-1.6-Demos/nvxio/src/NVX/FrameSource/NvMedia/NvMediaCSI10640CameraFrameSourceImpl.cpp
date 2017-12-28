/*
# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

#ifdef USE_CSI_OV10640

#include "Private/LogUtils.hpp"

#include "FrameSource/EGLAPIAccessors.hpp"

#include <NVX/FrameSource.hpp>
#include <NVX/Application.hpp>
#include <NVX/ConfigParser.hpp>
#include <NVX/ProfilerRange.hpp>

#include <cuda_runtime_api.h>
#include <cstring>

#include "FrameSource/NvMedia/NvMediaCSI10640CameraFrameSourceImpl.hpp"


using namespace nvidiaio::egl_api;

namespace nvidiaio
{

NvMediaCSI10640CameraFrameSourceImpl::NvMediaCSI10640CameraFrameSourceImpl(const std::string & configName, int number) :
    FrameSource(nvxio::FrameSource::CAMERA_SOURCE, "NvMediaCSI10640CameraFrameSource"),
    vxContext()
{
    nv12Frame = nullptr;

    ctx = nullptr;
    interopCtx = nullptr;

    cameraNumber = number;
    configPath = configName;
}

std::string NvMediaCSI10640CameraFrameSourceImpl::parseCameraConfig(const std::string& cameraConfigFile,
    CaptureConfigParams& captureConfigCollection)
{
    std::unique_ptr<nvxio::ConfigParser> cameraConfigParser(nvxio::createConfigParser());

    captureConfigCollection.i2cDevice = -1;
    captureConfigCollection.csiLanes = 2;

    cameraConfigParser->addParameter("capture-name", nvxio::OptionHandler::string(&captureConfigCollection.name));
    cameraConfigParser->addParameter("capture-description", nvxio::OptionHandler::string(&captureConfigCollection.description));
    cameraConfigParser->addParameter("board", nvxio::OptionHandler::string(&captureConfigCollection.board));
    cameraConfigParser->addParameter("input_device", nvxio::OptionHandler::string(&captureConfigCollection.inputDevice));
    cameraConfigParser->addParameter("input_format", nvxio::OptionHandler::string(&captureConfigCollection.inputFormat));
    cameraConfigParser->addParameter("surface_format", nvxio::OptionHandler::string(&captureConfigCollection.surfaceFormat));
    cameraConfigParser->addParameter("resolution", nvxio::OptionHandler::string(&captureConfigCollection.resolution));
    cameraConfigParser->addParameter("csi_lanes", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.csiLanes));
    cameraConfigParser->addParameter("interface", nvxio::OptionHandler::string(&captureConfigCollection.interface));
    cameraConfigParser->addParameter("embedded_lines_top", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.embeddedDataLinesTop));
    cameraConfigParser->addParameter("embedded_lines_bottom", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.embeddedDataLinesBottom));
    cameraConfigParser->addParameter("i2c_device", nvxio::OptionHandler::integer(&captureConfigCollection.i2cDevice));
    cameraConfigParser->addParameter("deserializer_address", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.desAddr));

    memset(captureConfigCollection.serAddr, 0, sizeof(captureConfigCollection.serAddr));
    memset(captureConfigCollection.sensorAddr, 0, sizeof(captureConfigCollection.sensorAddr));

    cameraConfigParser->addParameter("serializer_address", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.brdcstSerAddr));
    cameraConfigParser->addParameter("max9271_address_0", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.serAddr[0]));
    cameraConfigParser->addParameter("max9271_address_1", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.serAddr[1]));
    cameraConfigParser->addParameter("max9271_address_2", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.serAddr[2]));
    cameraConfigParser->addParameter("max9271_address_3", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.serAddr[3]));

    captureConfigCollection.brdcstSensorAddr = 0x30;
    cameraConfigParser->addParameter("sensor_address", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.brdcstSensorAddr));
    cameraConfigParser->addParameter("sensor_address_0", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.sensorAddr[0]));
    cameraConfigParser->addParameter("sensor_address_1", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.sensorAddr[1]));
    cameraConfigParser->addParameter("sensor_address_2", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.sensorAddr[2]));
    cameraConfigParser->addParameter("sensor_address_3", nvxio::OptionHandler::unsignedInteger(&captureConfigCollection.sensorAddr[3]));

    return cameraConfigParser->parse(cameraConfigFile);
}

bool NvMediaCSI10640CameraFrameSourceImpl::open()
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "FrameSource::open (NVXIO)");

    close();

    std::map<std::string, CaptureConfigParams>::iterator conf = cameraConfigCollection.find(configPath);
    if (conf != cameraConfigCollection.end())
    {
        captureConfigCollection = conf->second;
        NVXIO_PRINT("Prebuilt camera config from config preset is used ...");
    }
    else
    {
        nvxio::Application &app = nvxio::Application::get();
        std::string cameraConfigFile = app.findSampleFilePath("nvxio/cameras/" + configPath + ".ini");

        std::string message = parseCameraConfig(cameraConfigFile, captureConfigCollection);
        if (!message.empty())
        {
            NVXIO_PRINT("Error: %s", message.c_str());
            return false;
        }
    }

    if (cudaFree(nullptr) != cudaSuccess)
    {
        NVXIO_PRINT("Error: Failed to initialize CUDA context");
        return false;
    }

    // allocate objects
    ctx = new IPPCtx;
    ctx->imagesNum = cameraNumber;
    ctx->ippManager = nullptr;
    ctx->extImgDevice = nullptr;
    ctx->device = nullptr;
    std::memset(ctx->ipp, 0, sizeof(NvMediaIPPPipeline *) * NVMEDIA_MAX_PIPELINES_PER_MANAGER);

    interopCtx = new InteropContext;
    std::memset(interopCtx, 0, sizeof(InteropContext));
    interopCtx->producerExited = NVMEDIA_TRUE;

    if (IsFailed(IPPInit(ctx, captureConfigCollection)))
    {
        NVXIO_PRINT("Error: Failed to Initialize IPPInit");
        close();
        return false;
    }

    if (IsFailed(InteropInit(interopCtx, ctx)))
    {
        NVXIO_PRINT("Error: Failed to Initialize InteropInit");
        close();
        return false;
    }

    if(IsFailed(InteropProc(interopCtx)))
    {
        NVXIO_PRINT("Error: Failed to start InteropProc");
        close();
        return false;
    }

    if(IsFailed(IPPStart(ctx)))
    {
        NVXIO_PRINT("Error: Failed to start IPPStart");
        close();
        return false;
    }

    // fill frame source configuration
    configuration.frameWidth = ctx->inputWidth;
    configuration.frameHeight = ctx->inputHeight;
    configuration.fps = 30u;
    configuration.format = NVXCU_DF_IMAGE_NV12;

    nv12Frame = vxCreateImage(vxContext, ctx->inputWidth, ctx->inputHeight, VX_DF_IMAGE_NV12);
    NVXIO_CHECK_REFERENCE(nv12Frame);

    return true;
}

vx_image wrapNVXIOImage(vx_context context,
                        const image_t & image);

FrameSource::FrameStatus NvMediaCSI10640CameraFrameSourceImpl::fetch(const image_t & image,
                                                                     uint32_t /* timeout milliseconds*/)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "FrameSource::fetch (NVXIO)");

    if (!ctx || ctx->quit)
    {
        close();
        return nvxio::FrameSource::FrameStatus::CLOSED;
    }

    if (cudaSuccess != cudaFree(nullptr))
    {
        NVXIO_PRINT("Failed to initialize CUDA context");

        return nvxio::FrameSource::CLOSED;
    }

    vx_image image_ref = wrapNVXIOImage(vxContext, image);

    nvxcu_df_image_e frameFormat = image.format;
    vx_image workingImage = frameFormat == NVXCU_DF_IMAGE_NV12 ? image_ref : nv12Frame;

    for (vx_uint32 i = 0; i < ctx->imagesNum; ++i)
    {
        CUgraphicsResource cudaResource = nullptr;

        // Check for new frames in EglStream
        EGLint streamState = 0;

        for ( ; ; )
        {
            if (!eglQueryStreamKHR(ctx->eglDisplay, ctx->eglStream[i], EGL_STREAM_STATE_KHR, &streamState))
            {
                NVXIO_PRINT("Cuda consumer, eglQueryStreamKHR EGL_STREAM_STATE_KHR failed");
                close();

                return nvxio::FrameSource::FrameStatus::CLOSED;
            }

            if (streamState == EGL_STREAM_STATE_DISCONNECTED_KHR)
            {
                NVXIO_PRINT("CUDA Consumer: - EGL_STREAM_STATE_DISCONNECTED_KHR received");
                close();

                return nvxio::FrameSource::FrameStatus::CLOSED;
            }

            if (streamState == EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR)
                break;

            usleep(1000);
        }

        // Acquire frame in CUDA resource from EGLStream
        NVXIO_ASSERT( cuEGLStreamConsumerAcquireFrame(ctx->cudaConnection + i, &cudaResource, nullptr, 33000) == CUDA_SUCCESS );

        // If frame is acquired succesfully get the mapped CuEglFrame from CUDA resource
        CUeglFrame cudaEgl;
        NVXIO_ASSERT( cuGraphicsResourceGetMappedEglFrame(&cudaEgl, cudaResource, 0, 0) == CUDA_SUCCESS );

        NVXIO_ASSERT(cudaEgl.frameType == CU_EGL_FRAME_TYPE_ARRAY);
        NVXIO_ASSERT(cudaEgl.cuFormat == CU_AD_FORMAT_UNSIGNED_INT8);
        NVXIO_ASSERT(cudaEgl.eglColorFormat == CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR);
        NVXIO_ASSERT(cudaEgl.planeCount == 2);

        NVXIO_ASSERT(cudaEgl.height == configuration.frameHeight);
        NVXIO_ASSERT(cudaEgl.width * ctx->imagesNum == configuration.frameWidth);

        {
            vx_rectangle_t rect = { };

            rect.start_x = cudaEgl.width * i;
            rect.end_x = cudaEgl.width * (i + 1);
            rect.start_y = 0;
            rect.end_y = cudaEgl.height;

            // copy the first plane y

            vx_imagepatch_addressing_t addr;
            void *ptr;
            vx_map_id map_id;
            NVXIO_SAFE_CALL( vxMapImagePatch(workingImage, &rect, 0, &map_id, &addr, &ptr, VX_WRITE_ONLY, NVX_MEMORY_TYPE_CUDA, 0) );

            cudaStream_t stream = nullptr;
            NVXIO_ASSERT( cudaMemcpy2DFromArrayAsync(ptr, addr.stride_y,
                                                     (const struct cudaArray *) cudaEgl.frame.pArray[0],
                                                     0, 0,
                                                     cudaEgl.width * sizeof(vx_uint8), addr.dim_y,
                                                     cudaMemcpyDeviceToDevice, stream) == cudaSuccess );

            NVXIO_SAFE_CALL( vxUnmapImagePatch(workingImage, map_id) );

            // copy the second plane u/v

            vx_rectangle_t uv_rect = { rect.start_x >> 1, rect.start_y >> 1,
                                       rect.end_x   >> 1, rect.end_y   >> 1 };

            NVXIO_SAFE_CALL( vxMapImagePatch(workingImage, &uv_rect, 1, &map_id, &addr, &ptr, VX_WRITE_ONLY, NVX_MEMORY_TYPE_CUDA, 0) );

            NVXIO_ASSERT( (cudaMemcpy2DFromArrayAsync(ptr, addr.stride_y,
                                                      (const struct cudaArray *)cudaEgl.frame.pArray[1],
                                                      0, 0,
                                                      (cudaEgl.width >> 1) * sizeof(vx_uint16), addr.dim_y >> 1,
                                                      cudaMemcpyDeviceToDevice, stream) == cudaSuccess) );

            NVXIO_SAFE_CALL( vxUnmapImagePatch(workingImage, map_id) );

            NVXIO_ASSERT( cudaStreamSynchronize(stream) == cudaSuccess );
        }

        NVXIO_ASSERT( cuEGLStreamConsumerReleaseFrame(ctx->cudaConnection + i, cudaResource, nullptr) == CUDA_SUCCESS );
    }

    // copy or convert to output image
    if (frameFormat != NVXCU_DF_IMAGE_NV12)
    {
        NVXIO_SAFE_CALL( vxuColorConvert(vxContext, nv12Frame, image_ref) );
    }

    NVXIO_SAFE_CALL( vxReleaseImage(&image_ref) );

    return nvxio::FrameSource::FrameStatus::OK;
}

FrameSource::Parameters NvMediaCSI10640CameraFrameSourceImpl::getConfiguration()
{
    return configuration;
}

bool NvMediaCSI10640CameraFrameSourceImpl::setConfiguration(const FrameSource::Parameters& params)
{
    NVXIO_ASSERT(!ctx || ctx->quit);

    bool result = true;

    // ignore FPS, width, height values
    if (params.frameWidth != (uint32_t)-1)
        result = false;
    if (params.frameHeight != (uint32_t)-1)
        result = false;
    if (params.fps != (uint32_t)-1)
        result = false;

    NVXIO_ASSERT((params.format == NVXCU_DF_IMAGE_NV12) ||
                 (params.format == NVXCU_DF_IMAGE_U8) ||
                 (params.format == NVXCU_DF_IMAGE_RGB) ||
                 (params.format == NVXCU_DF_IMAGE_RGBX)||
                 (params.format == NVXCU_DF_IMAGE_NONE));

    configuration.format = params.format;

    return result;
}

void NvMediaCSI10640CameraFrameSourceImpl::close()
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "FrameSource::close (NVXIO)");

    if (nv12Frame)
    {
        vxReleaseImage(&nv12Frame);
        nv12Frame = nullptr;
    }

    if (ctx)
    {
        ctx->quit = NVMEDIA_TRUE;

        if (IsFailed(IPPStop(ctx)))
        {
            NVXIO_PRINT("Error: Failed to stop IPPStop");
        }

        if (interopCtx)
        {
            InteropFini(interopCtx);

            delete interopCtx;
            interopCtx = nullptr;
        }

        IPPFini(ctx);

        delete ctx;
        ctx = nullptr;
    }
}

NvMediaCSI10640CameraFrameSourceImpl::~NvMediaCSI10640CameraFrameSourceImpl()
{
    close();
}

} // namespace nvidiaio

#endif // USE_CSI_OV10640
