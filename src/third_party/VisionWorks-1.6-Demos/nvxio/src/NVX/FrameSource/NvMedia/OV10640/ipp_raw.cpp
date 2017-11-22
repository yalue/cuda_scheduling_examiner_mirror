/* Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifdef USE_CSI_OV10640

#include <limits.h>
#include <math.h>
#include <sys/time.h>

#include "board_name.h"

#include "buffer_utils.h"
#include "ipp_raw.hpp"
#include "isp_settings.h"
#include "misc_utils.h"
#include "ipp_component.hpp"

#include <string>
#include <iostream>

#include <NVX/Application.hpp>

//
// This is the callback function to get the global time
//
static NvMediaStatus
IPPGetAbsoluteGlobalTime(
    void *clientContext,
    NvMediaGlobalTime *timeValue)
{
    struct timeval tv;

    if(!timeValue)
        return NVMEDIA_STATUS_ERROR;

    gettimeofday(&tv, NULL);

    // Convert timeval to 64-bit microseconds
    *timeValue = (NvU64)tv.tv_sec * 1000000 + (NvU64)tv.tv_usec;

    return NVMEDIA_STATUS_OK;
}

// Event callback function
// This function will be called by IPP in case of an event
static void
IPPEventHandler(
    void *clientContext,
    NvMediaIPPComponentType componentType,
    NvMediaIPPComponent *ippComponent,
    NvMediaIPPEventType etype,
    NvMediaIPPEventData *edata)
{
    switch (etype) {
        case NVMEDIA_IPP_EVENT_INFO_FRAME_CAPTURE:
        {
            LOG_INFO("Image capture event\n");
            break;
        }
        case NVMEDIA_IPP_EVENT_WARNING_CAPTURE_FRAME_DROP:
        {
            LOG_WARN("Capture frame drop\n");
            break;
        }
        case NVMEDIA_IPP_EVENT_ERROR_NO_RESOURCES:
        {
            LOG_ERR("Out of resource\n");
            break;
        }
        case NVMEDIA_IPP_EVENT_ERROR_INTERNAL_FAILURE:
        {
            LOG_ERR("Internal failure\n");
            break;
        }
        case NVMEDIA_IPP_EVENT_ERROR_BUFFER_PROCESSING_FAILURE:
        {
            LOG_ERR("Buffer processing failure\n");
            break;
        }
        default:
        {
            break;
        }
    }
}

NvMediaStatus
IPPInit (
    IPPCtx *ctx,
    const CaptureConfigParams & captureParams)
{
    if (nvxio::Application::get().getVerboseFlag())
        SetLogLevel(LEVEL_DBG);

    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    NvU32 i;
    ExtImgDevParam extImgDevParam;

    if (!ctx) {
        LOG_ERR("%s: Bad parameter", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    memset(ctx->ippComponents, 0, sizeof(ctx->ippComponents));
    memset(ctx->ippIspComponents, 0, sizeof(ctx->ippIspComponents));
    memset(ctx->ippIscComponents, 0, sizeof(ctx->ippIscComponents));
    memset(ctx->ippControlAlgorithmComponents, 0, sizeof(ctx->ippControlAlgorithmComponents));
    memset(ctx->componentNum, 0, sizeof(ctx->componentNum));

    if (ctx->imagesNum > NVMEDIA_MAX_AGGREGATE_IMAGES) {
        LOG_WARN("Max aggregate images is: %u\n",
                 NVMEDIA_MAX_AGGREGATE_IMAGES);
        ctx->imagesNum = NVMEDIA_MAX_AGGREGATE_IMAGES;
    }

    ctx->quit = NVMEDIA_FALSE;

    ctx->aggregateFlag = NVMEDIA_TRUE;
    ctx->outputSurfType = NvMediaSurfaceType_Image_YUV_420;
    ctx->ispOutType = NvMediaSurfaceType_Image_YUV_420;
    ctx->inputFormatWidthMultiplier = 1;
    ctx->showTimeStamp = NVMEDIA_FALSE;
    ctx->showMetadataFlag = NVMEDIA_FALSE;
    ctx->pluginFlag = NVMEDIA_NVACPLUGIN;
    ctx->ippManager = NULL;

    ctx->camMap.enable = EXTIMGDEV_MAP_N_TO_ENABLE(ctx->imagesNum);
    ctx->camMap.mask = CAM_MASK_DEFAULT;
    ctx->camMap.csiOut = CSI_OUT_DEFAULT;

    LOG_DBG("%s: input resolution %s\n", __func__, captureParams.resolution.c_str());
    if (sscanf(captureParams.resolution.c_str(), "%ux%u",
        &ctx->inputWidth,
        &ctx->inputHeight) != 2) {
        LOG_ERR("%s: Invalid input resolution %s\n", __func__,
                captureParams.resolution.c_str());
        goto failed;
    }
    LOG_DBG("%s: inputWidth =%d,ctx->inputHeight =%d\n", __func__,
                ctx->inputWidth, ctx->inputHeight);

     if (ctx->aggregateFlag)
       ctx->inputWidth *= ctx->imagesNum;

    if (ctx->aggregateFlag &&
       (ctx->inputWidth % ctx->imagesNum) != 0) {
        LOG_ERR("%s: Invalid number of siblings (%u) for input width: %u\n",
                __func__, ctx->imagesNum, ctx->inputWidth);
        goto failed;
    }

    ctx->ippNum = ctx->imagesNum;
    for (i=0; i<ctx->ippNum; i++) {
        ctx->ispEnabled[i] = NVMEDIA_TRUE;
        ctx->outputEnabled[i] = NVMEDIA_TRUE;
        ctx->controlAlgorithmEnabled[i] = NVMEDIA_TRUE;
    }

    memset(&extImgDevParam, 0, sizeof(extImgDevParam));
    extImgDevParam.desAddr = captureParams.desAddr;
    extImgDevParam.brdcstSerAddr = captureParams.brdcstSerAddr;
    extImgDevParam.brdcstSensorAddr = captureParams.brdcstSensorAddr;
    for(i = 0; i < MAX_AGGREGATE_IMAGES; i++) {
        //FIXME: Hardcoded now, would be read from config file later
        //extImgDevParam.serAddr[i] = captureParams.brdcstSerAddr + i + 1;
        extImgDevParam.sensorAddr[i] = captureParams.brdcstSensorAddr + i + 1;
        //extImgDevParam.serAddr[i] = captureParams.serAddr[i];
        //extImgDevParam.sensorAddr[i] = captureParams.sensorAddr[i];
    }
    extImgDevParam.i2cDevice = captureParams.i2cDevice;
    extImgDevParam.moduleName = (char *)captureParams.inputDevice.c_str();
    extImgDevParam.board = (char *)captureParams.board.c_str();
    extImgDevParam.resolution = (char *)captureParams.resolution.c_str();
    extImgDevParam.sensorsNum = ctx->imagesNum;
    extImgDevParam.inputFormat = (char *)captureParams.inputFormat.c_str();
    extImgDevParam.interface = (char *)captureParams.interface.c_str();
    extImgDevParam.camMap = &ctx->camMap;
    extImgDevParam.enableEmbLines =
        (captureParams.embeddedDataLinesTop || captureParams.embeddedDataLinesBottom) ?
            NVMEDIA_TRUE : NVMEDIA_FALSE;
    extImgDevParam.initialized = NVMEDIA_FALSE;
    extImgDevParam.enableSimulator = NVMEDIA_FALSE;

    ctx->extImgDevice = ExtImgDevInit(&extImgDevParam);
    if(!ctx->extImgDevice) {
        LOG_ERR("%s: Failed to initialize ISC devices\n", __func__);
        status = NVMEDIA_STATUS_ERROR;
        goto failed;
    }

    ctx->captureEnabled = NVMEDIA_TRUE;
    if (ctx->captureEnabled) {
        if(IsFailed(IPPSetCaptureSettings(ctx, (CaptureConfigParams *)&captureParams)))
            goto failed;
    }

    ctx->device = NvMediaDeviceCreate();
    if(!ctx->device) {
        LOG_ERR("%s: Failed to create NvMedia device\n", __func__);
        goto failed;
    }

    // Create IPPManager
    ctx->ippManager = NvMediaIPPManagerCreate(NVMEDIA_IPP_VERSION_INFO, ctx->device, ctx->ipaDevice);
    if(!ctx->ippManager) {
        LOG_ERR("%s: Failed to create ippManager\n", __func__);
        goto done;
    }

    status = NvMediaIPPManagerSetTimeSource(ctx->ippManager, NULL, IPPGetAbsoluteGlobalTime);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: Failed to set time source\n", __func__);
        goto failed;
    }

    status = IPPCreateRawPipeline(ctx);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: Failed to create Raw Pipeline \n", __func__);
        goto failed;
    }

        // Set the callback function for event
    status = NvMediaIPPManagerSetEventCallback(ctx->ippManager, ctx, IPPEventHandler);
    if (status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: Failed to set event callback\n", __func__);
        return status;
    }

    return NVMEDIA_STATUS_OK;

failed:
    LOG_ERR("%s: Failed", __func__);
    IPPFini(ctx);
    return NVMEDIA_STATUS_ERROR;

done:
    return status;
}

NvMediaStatus
IPPStart (
    IPPCtx *ctx)
{
    NvU32 i;

    // Start IPPs
    for (i=0; i<ctx->ippNum; i++) {
        if (IsFailed(NvMediaIPPPipelineStart(ctx->ipp[i]))) {      //ippPipeline
            LOG_ERR("%s: Failed starting pipeline %d\n", __func__, i);
            goto failed;
        }
    }

#if BOARD_CODE_NAME >= DRIVE_PX2_BOARD
    // Start ExtImgDevice
    if(ctx->extImgDevice)
        ExtImgDevStart(ctx->extImgDevice);
#endif

    return NVMEDIA_STATUS_OK;

failed:
    LOG_ERR("%s: Failed", __func__);
    IPPFini(ctx);

    return NVMEDIA_STATUS_ERROR;
}
NvMediaStatus
IPPStop (IPPCtx *ctx)
{
    if (!ctx)
        return NVMEDIA_STATUS_OK;

    NvU32 i;

    for(i = 0; i < ctx->ippNum; i++) {
        if (!ctx->ipp[i])
            continue;

        if (IsFailed(NvMediaIPPPipelineStop(ctx->ipp[i])))
        {
            LOG_ERR("%s: Failed stop pipeline %d\n", __func__, i);
            return NVMEDIA_STATUS_ERROR;
        }

        ctx->ipp[i] = NULL;
    }
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
IPPFini (IPPCtx *ctx)
{
    if (!ctx)
        return NVMEDIA_STATUS_OK;

    ctx->quit = NVMEDIA_TRUE;

    if (ctx->ippManager)
    {
        NvMediaIPPManagerDestroy(ctx->ippManager);
        ctx->ippManager = NULL;
    }

    if(ctx->extImgDevice)
    {
        ExtImgDevDeinit(ctx->extImgDevice);
        ctx->extImgDevice = NULL;
    }

    if(ctx->device)
    {
        NvMediaDeviceDestroy(ctx->device);
        ctx->device = NULL;
    }

    return NVMEDIA_STATUS_OK;
}

#endif // USE_CSI_OV10640
