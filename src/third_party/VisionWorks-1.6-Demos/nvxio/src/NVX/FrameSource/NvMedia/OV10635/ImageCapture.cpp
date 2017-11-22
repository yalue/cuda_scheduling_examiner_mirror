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

#include "FrameSource/NvMedia/OV10635/ImageCapture.hpp"
#include "config_capture.h"

#include <NVX/Application.hpp>

#define IMGCAPTURE_BUFFERPOOL_SIZE          20
#define TIMEOUT 100

using namespace nvidiaio::egl_api;

//
// EGL context and Stream initialization routings
//

static bool InitializeEGLDisplay(nvidiaio::ov10635::ImgCapture & ctx)
{
    // Obtain the EGL display
    ctx.eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (ctx.eglDisplay == EGL_NO_DISPLAY)
    {
        printf("EGL failed to obtain display.\n");
        return false;
    }

    // Initialize EGL
    EGLBoolean eglStatus = eglInitialize(ctx.eglDisplay, 0, 0);
    if (!eglStatus)
    {
        printf("EGL failed to initialize.\n");
        return false;
    }

    return true;
}

static bool InitializeEGLStream(nvidiaio::ov10635::ImgCapture & ctx)
{
    static const EGLint streamAttr[] = { EGL_NONE };
    EGLint fifo_length = 4, latency = 0, timeout = 0, error = 0;

    if(!setupEGLExtensions())
    {
        printf("%s: eglSetupExtensions failed\n", __func__);
        return false;
    }

    ctx.eglStream = eglCreateStreamKHR(ctx.eglDisplay, streamAttr);
    if(ctx.eglStream == EGL_NO_STREAM_KHR)
    {
        error = eglGetError();
        printf("%s: Failed to create egl stream, error: %d\n", __func__, error);
        return false;
    }

    // Set stream attribute
    if(!eglStreamAttribKHR(ctx.eglDisplay, ctx.eglStream, EGL_CONSUMER_LATENCY_USEC_KHR, 16000))
    {
        printf("Consumer: streamAttribKHR EGL_CONSUMER_LATENCY_USEC_KHR failed\n");
    }
    if(!eglStreamAttribKHR(ctx.eglDisplay, ctx.eglStream, EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR, 16000))
    {
        printf("Consumer: streamAttribKHR EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR failed\n");
    }

    // Get stream attributes
    if(!eglQueryStreamKHR(ctx.eglDisplay, ctx.eglStream, EGL_STREAM_FIFO_LENGTH_KHR, &fifo_length))
    {
        printf("Consumer: eglQueryStreamKHR EGL_STREAM_FIFO_LENGTH_KHR failed\n");
    }
    if(!eglQueryStreamKHR(ctx.eglDisplay, ctx.eglStream, EGL_CONSUMER_LATENCY_USEC_KHR, &latency))
    {
        printf("Consumer: eglQueryStreamKHR EGL_CONSUMER_LATENCY_USEC_KHR failed\n");
    }
    if(!eglQueryStreamKHR(ctx.eglDisplay, ctx.eglStream, EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR, &timeout))
    {
        printf("Consumer: eglQueryStreamKHR EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR failed\n");
    }

    // Create EGL stream consumer
    CUresult curesult = cuEGLStreamConsumerConnect(&ctx.cudaConnection, ctx.eglStream);
    if (CUDA_SUCCESS != curesult)
    {
        printf("Connect CUDA EGL stream consumer ERROR %d\n", curesult);
        return false;
    }

    // Create EGL stream producer
    ctx.eglProducer = NvMediaEglStreamProducerCreate(ctx.device,
                                                     ctx.eglDisplay,
                                                     ctx.eglStream,
                                                     ctx.outputSurfType,
                                                     ctx.outputWidth,
                                                     ctx.outputHeight);
    if(!ctx.eglProducer)
    {
        printf("%s: Failed to create EGL stream, producer\n", __func__);
        return false;
    }

    return true;
}

static void FinalizeEglDisplay(nvidiaio::ov10635::ImgCapture & ctx)
{
    eglTerminate(ctx.eglDisplay);
    ctx.eglDisplay = EGL_NO_DISPLAY;
}

static void FinalizeEglStream(nvidiaio::ov10635::ImgCapture & ctx)
{
    if(ctx.eglProducer)
    {
        NvMediaEglStreamProducerDestroy(ctx.eglProducer);
        ctx.eglProducer = NULL;
    }

    cuEGLStreamConsumerDisconnect(&ctx.cudaConnection);
    ctx.cudaConnection = NULL;

    if(ctx.eglStream)
    {
        eglDestroyStreamKHR(ctx.eglDisplay, ctx.eglStream);
        ctx.eglStream = EGL_NO_STREAM_KHR;
    }
}

static NvU32
ImgCapture_displayThreadFunc(void *data)
{
    nvidiaio::ov10635::ImgCapture *ctx = (nvidiaio::ov10635::ImgCapture *)data;
    NvMediaStatus status;
    NvMediaImage *capturedFrame = NULL;
    NvMediaImage *imgOut = NULL;
    NvU32 i = 0, numFrames = 0, timeout = TIMEOUT;

    while(!ctx->quit)
    {
        // Wait for captured frames
        status = ImageCaptureNumAvailableFrames(ctx->icpCtx, &numFrames);
        if(status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("%s: ImageCaptureNumAvailableFrames failed\n", __func__);
            ctx->quit = NVMEDIA_TRUE;
            goto done;
        }

        if(!numFrames)
        {
            usleep(100);
            continue;
        }

        // Get frame from capture
        status = ImageCaptureGetFrame(ctx->icpCtx, &capturedFrame);
        if(status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("%s: ImageCaptureGetFrame failed\n", __func__);
            ctx->quit = NVMEDIA_TRUE;
            goto done;
        }

        // Convert frame and push to EGLStream
        {
            NVXIO_ASSERT(ctx->convert);

            // Setup image for conversion
            status = ImageSurfUtilsConvertImage(ctx->convert, capturedFrame);
            if(status != NVMEDIA_STATUS_OK)
            {
                LOG_ERR("%s: ImageSurfUtilsConvertImage failed\n", __func__);
                ctx->quit = NVMEDIA_TRUE;
                goto done;
            }

#if BOARD_CODE_NAME >= DRIVE_PX_BOARD
            capturedFrame = NULL;
#endif

            timeout = TIMEOUT;
            numFrames = 0;
            // Check for converted image
            while((!numFrames) && timeout--)
            {
                status = ImageSurfUtilsNumAvailableFrames(ctx->convert,
                                                          &numFrames);
                if(status != NVMEDIA_STATUS_OK)
                {
                    LOG_ERR("%s: ImageCaptureNumAvailableFrames failed\n", __func__);
                    ctx->quit = NVMEDIA_TRUE;
                    goto done;
                }

                const timespec time = {0, 50000};
                nanosleep(&time, NULL);
            }

            if(numFrames)
            {
                // Get image from conversion
                status = ImageSurfUtilsGetImage(ctx->convert, &imgOut);
                if(status != NVMEDIA_STATUS_OK)
                {
                    LOG_ERR("%s: ImageSurfUtilsGetImage failed\n", __func__);
                    ctx->quit = NVMEDIA_TRUE;
                    goto done;
                }

                // push to EGLStream stuff
                {
                    if (!ctx->quit)
                    {
                        NvMediaImage * imageToPass = imgOut, * retImage = NULL;

                        LOG_DBG("%s: EGL producer: Post image %p\n", __func__, imageToPass);
                        if(NvMediaEglStreamProducerPostImage(ctx->eglProducer,
                                                             imageToPass,
                                                             NULL) != NVMEDIA_STATUS_OK)
                        {
                            printf("%s: NvMediaEglStreamProducerPostImage failed\n", __func__);
                            ctx->quit = NVMEDIA_TRUE;
                            break;
                        }

                        LOG_DBG("%s: EGL producer Getting image %p\n", __func__, retImage);
                        NvMediaEglStreamProducerGetImage(ctx->eglProducer, &retImage, 100);

                        if(retImage)
                        {
                            LOG_DBG("%s: EGL producer: Got image %p\n", __func__, retImage);

                            // Put post-processed buffer on output queue
                            ImageSurfUtilsReleaseImage(ctx->convert, retImage);
                        }

                        LOG_DBG("capture2d_AggrPostProcessorThreadFunc iteraration finished.\n");
                    }

                    goto done;
                }
            }
        }
done:
        if(capturedFrame)
            ImageCaptureReleaseImage(capturedFrame);
        i++;

        if(ctx->quit)
            break;
    }

    ctx->displayThread.exitedFlag = NVMEDIA_TRUE;

    return 0;
}

static NvMediaStatus
ImgCapture_SetConfigCaptureParameters(nvidiaio::ov10635::ImgCapture *ctx,
                                      NvMediaBool aggregateFlag,
                                      ConfigCaptureParameters *param)
{
    if (!ctx || !param)
    {
        LOG_ERR("%s: Bad parameter passed\n");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    memset(param, 0, sizeof(ConfigCaptureParameters));
    param->inputFormat = (char *)ctx->captureParams->inputFormat.c_str();
    param->surfaceFormat = (char *)ctx->captureParams->surfaceFormat.c_str();
    param->resolution = (char *)ctx->captureParams->resolution.c_str();
    param->interface = (char *)ctx->captureParams->interface.c_str();
    param->csiLanes= ctx->captureParams->csiLanes;
    param->embeddedDataLinesTop = ctx->captureParams->embeddedDataLinesTop;
    param->embeddedDataLinesBottom = ctx->captureParams->embeddedDataLinesBottom;
    param->device = ctx->device;
    param->aggregateFlag = aggregateFlag;
    param->imagesNum = ctx->imagesNum;

#if BOARD_CODE_NAME >= DRIVE_PX_BOARD
    param->pixelOrder = (ctx->tpgMode) ? NVMEDIA_RAW_PIXEL_ORDER_RGGB :
                                         ctx->extImgDevice->property.pixelOrder;
#endif

    if(param->surfaceFormat[0] == '\0')
        param->isSurfaceFormatUsed = NVMEDIA_FALSE;
    else
        param->isSurfaceFormatUsed = NVMEDIA_TRUE;

    return NVMEDIA_STATUS_OK;
}

static void
ImgCapture_SetImageCaptureTestConfigSettings(nvidiaio::ov10635::ImgCapture *ctx,
                                             ImageCaptureTestConfig *icpTestConfig)
{
    memset(icpTestConfig, 0, sizeof(ImageCaptureTestConfig));

    icpTestConfig->numBuffers = IMGCAPTURE_BUFFERPOOL_SIZE;
    icpTestConfig->quitFlag = &ctx->quit;

#if BOARD_CODE_NAME >= DRIVE_PX_BOARD
    icpTestConfig->numMiniburstFrames = 1;
#endif
}

static void
ImgCapture_SetImageSurfUtilsParameters(nvidiaio::ov10635::ImgCapture *ctx,
                                       NvU32 inputWidth,
                                       NvU32 inputHeight,
                                       ImageSurfUtilsParameters *convertParam)
{
    memset(convertParam, 0, sizeof(ImageSurfUtilsParameters));

    convertParam->inputFormat = (char *)ctx->captureParams->inputFormat.c_str();
    convertParam->inputWidth = inputWidth;
    convertParam->inputHeight = inputHeight;
    convertParam->rawBytesPerPixel = ctx->rawBytesPerPixel;

#if BOARD_CODE_NAME >= DRIVE_PX_BOARD
    convertParam->pixelOrder = ctx->extImgDevice->property.pixelOrder;
#endif

    convertParam->quitFlag = &ctx->quit;

#if BOARD_CODE_NAME >= DRIVE_PX2_BOARD
    convertParam->bDisplay = NVMEDIA_TRUE;
#endif
}

#if BOARD_CODE_NAME == JETSON_PRO_BOARD

static NvMediaStatus
ImgCapture_SetConfigISCInfoParameters(nvidiaio::ov10635::ImgCapture *ctx)
{
    CaptureConfigParams *captureParams = ctx->captureParams;
    ConfigISCInfo *iscConfigInfo = &ctx->iscConfigInfo;

    iscConfigInfo->max9286_address = captureParams->desAddr;
    iscConfigInfo->broadcast_max9271_address = captureParams->brdcstSerAddr;
    iscConfigInfo->broadcast_sensor_address = captureParams->brdcstSensorAddr;

    for(unsigned int i = 0; i < MAX_AGGREGATE_IMAGES; i++)
    {
        iscConfigInfo->max9271_address[i] = captureParams->serAddr[i];
        iscConfigInfo->sensor_address[i] = captureParams->sensorAddr[i];
    }

    iscConfigInfo->i2cDevice = captureParams->i2cDevice;
    ctx->captureModuleName = (char *)captureParams->inputDevice.c_str();
    iscConfigInfo->board = (char *)captureParams->board.c_str();
    iscConfigInfo->resolution = (char *)captureParams->resolution.c_str();

    iscConfigInfo->sensorsNum = ctx->imagesNum;
    if(captureParams->inputFormat == "raw10")
        iscConfigInfo->rawCompressionFormat = ISC_RAW10;
    else if(captureParams->inputFormat == "raw12")
        iscConfigInfo->rawCompressionFormat = ISC_RAW1x12;
    else if(captureParams->inputFormat == "raw2x11")
        iscConfigInfo->rawCompressionFormat = ISC_RAW2x11;
    else if(captureParams->inputFormat == "raw16log")
        iscConfigInfo->rawCompressionFormat = ISC_RAW16LOG;

    return NVMEDIA_STATUS_OK;
}

#else

static NvMediaStatus
ImgCapture_SetExtImgDevParameters(nvidiaio::ov10635::ImgCapture *ctx, ExtImgDevParam *configParam)
{
    unsigned int i;
    CaptureConfigParams *captureParams = ctx->captureParams;

    configParam->desAddr = captureParams->desAddr;
    configParam->brdcstSerAddr = captureParams->brdcstSerAddr;
    configParam->brdcstSensorAddr = captureParams->brdcstSensorAddr;

    for(i = 0; i < MAX_AGGREGATE_IMAGES; i++)
    {
        configParam->serAddr[i] = captureParams->serAddr[i];
        configParam->sensorAddr[i] = captureParams->sensorAddr[i];
    }

    configParam->i2cDevice = captureParams->i2cDevice;
    configParam->moduleName = (char *)captureParams->inputDevice.c_str();
    configParam->board = (char *)captureParams->board.c_str();
    configParam->resolution = (char *)captureParams->resolution.c_str();
    configParam->camMap = &ctx->camMap;
    configParam->sensorsNum = ctx->imagesNum;
    configParam->inputFormat = (char *)captureParams->inputFormat.c_str();
    configParam->interface = (char *)captureParams->interface.c_str();
    configParam->enableEmbLines =
        (captureParams->embeddedDataLinesTop || captureParams->embeddedDataLinesBottom) ?
            NVMEDIA_TRUE : NVMEDIA_FALSE;
    configParam->initialized = NVMEDIA_FALSE;
    configParam->enableSimulator = NVMEDIA_FALSE;

    return NVMEDIA_STATUS_OK;
}

#endif

namespace nvidiaio { namespace ov10635 {

NvMediaStatus
ImgCapture_Init(ImgCapture **ctx, CaptureConfigParams & captureConfigCollection, NvU32 imagesNum)
{
    if (nvxio::Application::get().getVerboseFlag())
        SetLogLevel(LEVEL_DBG);

    NvMediaStatus status;
    ImgCapture *ctxTmp = nullptr;
    ImageCaptureTestConfig icpTestConfig;
    ImageSurfUtilsParameters convertParam;
    ConfigCaptureParameters configParam;
    ConfigCaptureSettings configSettings;
    NvMediaBool useAggregationFlag = NVMEDIA_TRUE;

#if BOARD_CODE_NAME >= DRIVE_PX_BOARD
    ExtImgDevParam extImgDevParam;
#endif

    NvU32 inputWidth = 0;
    NvU32 inputHeight = 0;

    memset(&configSettings, 0, sizeof(ConfigCaptureSettings));

    if(!ctx)
    {
        LOG_ERR("%s: Bad parameter", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    // Create and initialize ImgCapture context
    ctxTmp = (ImgCapture *)calloc(1, sizeof(ImgCapture));
    if(!ctxTmp)
    {
        LOG_ERR("%s: Out of memory", __func__);
        return NVMEDIA_STATUS_OUT_OF_MEMORY;
    }

    if(imagesNum > NVMEDIA_MAX_AGGREGATE_IMAGES)
    {
        LOG_WARN("Max aggregate images is: %u\n",
                 NVMEDIA_MAX_AGGREGATE_IMAGES);
        imagesNum = NVMEDIA_MAX_AGGREGATE_IMAGES;
    }

    ctxTmp->imagesNum = imagesNum;
    ctxTmp->displayThread.exitedFlag = NVMEDIA_TRUE;
    ctxTmp->captureParams = &captureConfigCollection;
    ctxTmp->tpgMode = NVMEDIA_FALSE;

#if BOARD_CODE_NAME >= DRIVE_PX_BOARD
    ctxTmp->camMap.enable = EXTIMGDEV_MAP_N_TO_ENABLE(imagesNum);
    ctxTmp->camMap.mask   = CAM_MASK_DEFAULT;
    ctxTmp->camMap.csiOut = CSI_OUT_DEFAULT;
#endif

    if (sscanf(ctxTmp->captureParams->resolution.c_str(), "%ux%u", &inputWidth, &inputHeight) != 2)
    {
        LOG_ERR("%s: Invalid input resolution %s\n", __func__, ctxTmp->captureParams->resolution.c_str());
        status = NVMEDIA_STATUS_ERROR;
        goto failed;
    }

    if (useAggregationFlag)
        inputWidth *= imagesNum;

    ctxTmp->outputWidth = inputWidth;
    ctxTmp->outputHeight = inputHeight;
    ctxTmp->outputSurfType = NvMediaSurfaceType_Image_RGBA;

    if (captureConfigCollection.inputFormat == "raw12")
    {
        ctxTmp->outputWidth >>= 1;
        ctxTmp->outputHeight >>= 1;
    }

    // create EGLDisplay
    if (!InitializeEGLDisplay(*ctxTmp))
    {
        status = NVMEDIA_STATUS_ERROR;
        LOG_ERR("%s: Failed to initialize EGLDisplay\n", __func__);
        goto failed;
    }

    // create EGLStream
    if (!InitializeEGLStream(*ctxTmp))
    {
        status = NVMEDIA_STATUS_ERROR;
        LOG_ERR("%s: Failed to initialize EGLStream\n", __func__);
        goto failed;
    }

#if BOARD_CODE_NAME > JETSON_PRO_BOARD

    extImgDevParam.slave = NVMEDIA_FALSE;
    status = ImgCapture_SetExtImgDevParameters(ctxTmp, &extImgDevParam);
    if(status != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("%s: Failed to set ISC device parameters\n", __func__);
        goto failed;
    }

    ctxTmp->extImgDevice = ExtImgDevInit(&extImgDevParam);
    if(!ctxTmp->extImgDevice)
    {
        LOG_ERR("%s: Failed to initialize ISC devices\n", __func__);
        status = NVMEDIA_STATUS_ERROR;
        goto failed;
    }

#endif

    // Create NvMedia Device
    ctxTmp->device = NvMediaDeviceCreate();
    if(!ctxTmp->device)
    {
        status = NVMEDIA_STATUS_ERROR;
        LOG_ERR("%s: Failed to create NvMedia device\n", __func__);
        goto failed;
    }

    // Configure and set capture settings
    status = ImgCapture_SetConfigCaptureParameters(ctxTmp,
                                                   useAggregationFlag,
                                                   &configParam);
    if(status != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("%s: Failed to set capture parameters\n", __func__);
        goto failed;
    }

    status = ConfigCaptureProcessParameters(&configParam,
                                            &configSettings);
    if(status != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("%s: Failed to process capture parameters\n", __func__);
        goto failed;
    }
    ctxTmp->rawBytesPerPixel = ConfigCaptureGetRawBytesPerPixel(&configSettings);
    ImgCapture_SetImageCaptureTestConfigSettings(ctxTmp, &icpTestConfig);

    // Initialize and create converter
    {
        ImgCapture_SetImageSurfUtilsParameters(ctxTmp,
                                               inputWidth,
                                               inputHeight,
                                               &convertParam);

        status = ImageSurfUtilsCreate(&ctxTmp->convert, ctxTmp->device,
                                      &convertParam);
        if(status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("%s: Failed to create ImageSurfUtils context\n", __func__);
            goto failed;
        }
    }

    // Configure and create ISC devices

#if BOARD_CODE_NAME == JETSON_PRO_BOARD

    status = ImgCapture_SetConfigISCInfoParameters(ctxTmp);
    if(status != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("%s: Failed to set ISC device parameters\n", __func__);
        goto failed;
    }

    ctxTmp->iscConfigInfo.inputSurfType = configSettings.poolConfig.surfType;
    ctxTmp->iscConfigInfo.csi_link = configSettings.settings.interfaceType;
    status = ConfigISCCreateDevices(&ctxTmp->iscConfigInfo,
                                    &ctxTmp->iscDevices,
                                    NVMEDIA_FALSE,
                                    ctxTmp->captureModuleName);
    if(status != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("%s: Failed to create ISC devices\n", __func__);
        goto failed;
    }

#endif

    // Create capture context and thread
    status = ImageCaptureCreate(&ctxTmp->icpCtx,
                                ConfigCaptureGetICPSettings(&configSettings),
                                ConfigCaptureGetBufferPoolConfig(&configSettings),
                                &icpTestConfig);
    if(status != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("%s: Failed to create ImageCapture Context\n", __func__);
        goto failed;
    }

    // Create thread to save and/or display frames
    ctxTmp->displayThread.exitedFlag = NVMEDIA_FALSE;
    status = NvThreadCreate(&ctxTmp->displayThread.thread,
                            ImgCapture_displayThreadFunc,
                            (void *)ctxTmp, NV_THREAD_PRIORITY_NORMAL);

    if(status != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("%s: Failed to create save and display thread\n",
                __func__);
        ctxTmp->displayThread.exitedFlag = NVMEDIA_TRUE;
        goto failed;
    }

#if BOARD_CODE_NAME >= DRIVE_PX2_BOARD
    // Start ExtImgDevice
    if(ctxTmp->extImgDevice)
        ExtImgDevStart(ctxTmp->extImgDevice);
#endif

    *ctx = ctxTmp;
    return NVMEDIA_STATUS_OK;
failed:
    LOG_ERR("%s: Failed to initialize ImgCapture\n",__func__);
    ImgCapture_Finish(ctxTmp);
    return status;
}

void
ImgCapture_Finish(ImgCapture *ctx)
{
    NvMediaStatus status;

    ctx->quit = NVMEDIA_TRUE;

    // Wait for all threads to exit
    while(!ctx->displayThread.exitedFlag)
    {
        LOG_DBG("%s: Waiting for save and display thread to quit\n",
                __func__);
        usleep(1000);
    }

    if(ctx->icpCtx)
        ImageCaptureDestroy(ctx->icpCtx);

    // Dispose converter
    if (ctx->convert)
        ImageSurfUtilsDestroy(ctx->convert);

    // Destroy threads
    if(ctx->displayThread.thread)
    {
        status = NvThreadDestroy(ctx->displayThread.thread);
        if(status != NVMEDIA_STATUS_OK)
            LOG_ERR("%s: Failed to destroy save and display thread\n", __func__);
    }

#if BOARD_CODE_NAME == JETSON_PRO_BOARD

    ConfigISCDestroyDevices(&ctx->iscConfigInfo, &ctx->iscDevices);

#else

    if (ctx->extImgDevice)
        ExtImgDevDeinit(ctx->extImgDevice);

#endif

    // Finalize EGL Stuff
    FinalizeEglStream(*ctx);
    FinalizeEglDisplay(*ctx);

    free(ctx);
}

} // namespace ov10635

} // namespace nvidiaio

#endif // USE_CSI_OV10635
