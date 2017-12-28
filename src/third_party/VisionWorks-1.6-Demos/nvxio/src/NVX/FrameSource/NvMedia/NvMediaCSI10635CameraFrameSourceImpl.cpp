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

#ifdef USE_CSI_OV10635

#include "Private/LogUtils.hpp"

#include "FrameSource/EGLAPIAccessors.hpp"

#include <NVX/FrameSource.hpp>
#include <NVX/Application.hpp>
#include <NVX/ConfigParser.hpp>
#include <NVX/ProfilerRange.hpp>

#include <cuda_runtime_api.h>

#include "FrameSource/NvMedia/NvMediaCSI10635CameraFrameSourceImpl.hpp"
#include "FrameSource/NvMedia/NvMediaCameraConfigParams.hpp"

using namespace nvidiaio::egl_api;

namespace nvidiaio
{

NvMediaCSI10635CameraFrameSourceImpl::NvMediaCSI10635CameraFrameSourceImpl(const std::string & configName, int number) :
    FrameSource(nvxio::FrameSource::CAMERA_SOURCE, "NvMediaCSI10635CameraFrameSource"),
    vxContext()
{
    context = nullptr;
    cameraNumber = number;
    configPath = configName;
}

std::string NvMediaCSI10635CameraFrameSourceImpl::parseCameraConfig(const std::string& cameraConfigFile,
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

bool NvMediaCSI10635CameraFrameSourceImpl::open()
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

    if (ov10635::ImgCapture_Init(&context, captureConfigCollection, cameraNumber) != NVMEDIA_STATUS_OK)
    {
        NVXIO_PRINT("Error: Failed to Initialize ImgCapture");
        return false;
    }

    // fill frame source configuration
    configuration.frameWidth = context->outputWidth;
    configuration.frameHeight = context->outputHeight;
    configuration.fps = 30;
    configuration.format = NVXCU_DF_IMAGE_RGBX;

    return true;
}

void convertFrame(const image_t & image,
                  const FrameSource::Parameters & configuration,
                  int width, int height,
                  bool usePitch, size_t pitch,
                  int depth, void * decodedPtr,
                  bool is_cuda,
                  void *& devMem,
                  size_t & devMemPitch);

vx_image wrapNVXIOImage(vx_context context,
                        const image_t & image);

FrameSource::FrameStatus NvMediaCSI10635CameraFrameSourceImpl::fetch(const image_t & image,
                                                                          uint32_t timeout /*milliseconds*/)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "FrameSource::fetch (NVXIO)");

    if (context->quit)
    {
        close();
        return nvxio::FrameSource::FrameStatus::CLOSED;
    }

    if (cudaSuccess != cudaFree(nullptr))
    {
        NVXIO_PRINT("Failed to initialize CUDA context");
        return nvxio::FrameSource::CLOSED;
    }

    CUresult cuStatus;
    CUgraphicsResource cudaResource;

    EGLint streamState = 0;
    if (!eglQueryStreamKHR(context->eglDisplay, context->eglStream, EGL_STREAM_STATE_KHR, &streamState))
    {
        NVXIO_PRINT("Cuda consumer, eglQueryStreamKHR EGL_STREAM_STATE_KHR failed");
        return nvxio::FrameSource::FrameStatus::CLOSED;
    }

    if (streamState == EGL_STREAM_STATE_DISCONNECTED_KHR)
    {
        NVXIO_PRINT("CUDA Consumer: - EGL_STREAM_STATE_DISCONNECTED_KHR received");
        return nvxio::FrameSource::FrameStatus::CLOSED;
    }

    if (streamState != EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR)
    {
        return nvxio::FrameSource::TIMEOUT;
    }

    cuStatus = cuEGLStreamConsumerAcquireFrame(&context->cudaConnection, &cudaResource, nullptr, timeout);
    if (cuStatus != CUDA_SUCCESS)
    {
        NVXIO_PRINT("Cuda Acquire failed cuStatus=%d", cuStatus);

        return nvxio::FrameSource::FrameStatus::TIMEOUT;
    }

    CUeglFrame eglFrame;
    cuStatus = cuGraphicsResourceGetMappedEglFrame(&eglFrame, cudaResource, 0, 0);
    if (cuStatus != CUDA_SUCCESS)
    {
        const char* error;
        cuGetErrorString(cuStatus, &error);
        NVXIO_PRINT("Cuda get resource failed with error: \"%s\"", error);
        return nvxio::FrameSource::FrameStatus::CLOSED;
    }

    NVXIO_ASSERT(eglFrame.width == configuration.frameWidth);
    NVXIO_ASSERT(eglFrame.height == configuration.frameHeight);

    NVXIO_ASSERT(configuration.format == NVXCU_DF_IMAGE_RGBX);
    NVXIO_ASSERT(eglFrame.pitch == ((eglFrame.width * 4 + 3) >> 2) << 2);
    convertFrame(image,
                 configuration,
                 eglFrame.width, eglFrame.height,
                 true, eglFrame.pitch,
                 4, eglFrame.frame.pPitch[0],
                 true,
                 devMem,
                 devMemPitch);
    NVXIO_ASSERT(devMem == nullptr && devMemPitch == 0);

    cuStatus = cuEGLStreamConsumerReleaseFrame(&context->cudaConnection, cudaResource, nullptr);
    if (cuStatus != CUDA_SUCCESS)
    {
        NVXIO_PRINT("Failed to release EGL frame");
        close();
        return nvxio::FrameSource::FrameStatus::CLOSED;
    }

    return nvxio::FrameSource::FrameStatus::OK;
}

FrameSource::Parameters NvMediaCSI10635CameraFrameSourceImpl::getConfiguration()
{
    return configuration;
}

bool NvMediaCSI10635CameraFrameSourceImpl::setConfiguration(const FrameSource::Parameters& params)
{
    NVXIO_ASSERT(!context || context->quit);

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

void NvMediaCSI10635CameraFrameSourceImpl::close()
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "FrameSource::close (NVXIO)");

    if (context)
    {
        ov10635::ImgCapture_Finish(context);
        context = nullptr;
    }
}

NvMediaCSI10635CameraFrameSourceImpl::~NvMediaCSI10635CameraFrameSourceImpl()
{
    close();
}

} // namespace nvidiaio

#endif // USE_CSI_OV10635
