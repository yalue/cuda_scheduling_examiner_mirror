/*
 * Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifdef USE_CSI_OV10640

#if __STDC_VERSION__ >= 199901L
#define _XOPEN_SOURCE 600
#else
#define _XOPEN_SOURCE 500
#endif /* __STDC_VERSION__ */

#include <stdlib.h>
#include <string.h>

#include "plugin.h"
#include "log_utils.h"

#define MIN(a,b) ((a < b) ? a : b)
#define CLIP(x, a, b)  (x > b?b:(x < a?a:x))
#define PRINT_ISPSTATS_FORDEBUG

static void
initializeLACSettings0(
    PluginContext *ctx,
    NvMediaIPPStreamType type,
    int width,
    int height,
    NvMediaISCSensorMode sensorMode)
{
    int numWindowsX, numWindowsY;
    int numPixels, numRemainder;
    float maxLac0Range;
    NvMediaIPPPluginOutputStreamSettings *streamSettings =
        &ctx->streamSettings[type];

    if(sensorMode == NVMEDIA_ISC_SENSOR_MODE_10BIT) {
        maxLac0Range = (2 ^ 10) / (2 ^ 14);
    }
    else {
        maxLac0Range = (2 ^ 12) / (2 ^ 14);
    }

    streamSettings->lacSettingsValid[0] = NVMEDIA_TRUE;
    if (ctx->ispVersion == NVMEDIA_IPP_ISP_VERSION_4) {
        int i, j, index = 0;
        int ROIOffset_x = 0, ROIOffset_y = 0;

        streamSettings->lacSettings[0].v4.enable = 1;
        streamSettings->lacSettings[0].v4.pixelFormat = NVMEDIA_ISP_PIXELFORMAT_QUAD;
        streamSettings->lacSettings[0].v4.rgbToYGain[0] = 0.29296875;
        streamSettings->lacSettings[0].v4.rgbToYGain[1] = 0.57421875;
        streamSettings->lacSettings[0].v4.rgbToYGain[2] = 0.132812545;
        streamSettings->lacSettings[0].v4.rgbToYGain[3] = 0;
        //Assuming the default sensor mode is 12bit
        streamSettings->lacSettings[0].v4.range[0].low = 0.0;
        streamSettings->lacSettings[0].v4.range[0].high = maxLac0Range;
        streamSettings->lacSettings[0].v4.range[1].low = 0.0;
        streamSettings->lacSettings[0].v4.range[1].high = maxLac0Range;
        streamSettings->lacSettings[0].v4.range[2].low = 0.0;
        streamSettings->lacSettings[0].v4.range[2].high = maxLac0Range;
        streamSettings->lacSettings[0].v4.range[3].low = 0.0;
        streamSettings->lacSettings[0].v4.range[3].high = maxLac0Range;
        //Set up 2x2 ROI grid
        for (i=0; i<2; i++, ROIOffset_y += height/2) {
            ROIOffset_x = 0;
            for (j=0; j<2; j++, ROIOffset_x += width/2, index++) {
                streamSettings->lacSettings[0].v4.ROIEnable[index] = NVMEDIA_TRUE;
                numWindowsX = 32;
                numPixels = width / (numWindowsX * 2);
                numRemainder = (width / 2)- (numPixels * numWindowsX);
                streamSettings->lacSettings[0].v4.windows[index].horizontalInterval = numPixels;
                streamSettings->lacSettings[0].v4.windows[index].size.width =
                            numPixels >= 32 ? 32 :
                            numPixels >= 16 ? 16 :
                            numPixels >= 8  ? 8  :
                            numPixels >= 4  ? 4  :
                            numPixels >= 2  ? 2  : 0;
                streamSettings->lacSettings[0].v4.windows[index].startOffset.x =
                    (numRemainder + (numPixels - streamSettings->lacSettings[0].v4.windows[index].size.width)) / 2 + ROIOffset_x;
                numWindowsY = 32;
                numPixels = height / (numWindowsY * 2);
                numRemainder = (height / 2) - (numPixels * numWindowsY);
                streamSettings->lacSettings[0].v4.windows[index].verticalInterval = numPixels;
                streamSettings->lacSettings[0].v4.windows[index].size.height =
                            numPixels >= 32 ? 32 :
                            numPixels >= 16 ? 16 :
                            numPixels >= 8  ? 8  :
                            numPixels >= 4  ? 4  :
                            numPixels >= 2  ? 2  : 0;
                streamSettings->lacSettings[0].v4.windows[index].startOffset.y =
                    (numRemainder + (numPixels - streamSettings->lacSettings[0].v4.windows[index].size.height)) / 2 + ROIOffset_y;
                streamSettings->lacSettings[0].v4.windows[index].horizontalNum = numWindowsX;
                streamSettings->lacSettings[0].v4.windows[index].verticalNum = numWindowsY;
            }
        }
    } else if (ctx->ispVersion == NVMEDIA_IPP_ISP_VERSION_3) {
        streamSettings->lacSettings[0].v3.enable = 1;
        streamSettings->lacSettings[0].v3.pixelFormat = NVMEDIA_ISP_PIXELFORMAT_QUAD;
        streamSettings->lacSettings[0].v3.rgbToYGain[0] = 0.29296875;
        streamSettings->lacSettings[0].v3.rgbToYGain[1] = 0.57421875;
        streamSettings->lacSettings[0].v3.rgbToYGain[2] = 0.132812545;
        streamSettings->lacSettings[0].v3.rgbToYGain[3] = 0;
        //Assuming the default sensor mode is 12bit
        streamSettings->lacSettings[0].v3.range[0].low = 0.0;
        streamSettings->lacSettings[0].v3.range[0].high = maxLac0Range;
        streamSettings->lacSettings[0].v3.range[1].low = 0.0;
        streamSettings->lacSettings[0].v3.range[1].high = maxLac0Range;
        streamSettings->lacSettings[0].v3.range[2].low = 0.0;
        streamSettings->lacSettings[0].v3.range[2].high = maxLac0Range;
        streamSettings->lacSettings[0].v3.range[3].low = 0.0;
        streamSettings->lacSettings[0].v3.range[3].high = maxLac0Range;

        numWindowsX = 64;
        numPixels = width / numWindowsX;
        numRemainder = width - (numPixels * numWindowsX);
        streamSettings->lacSettings[0].v3.windows.horizontalInterval = numPixels;
        streamSettings->lacSettings[0].v3.windows.size.width =
                    numPixels >= 32 ? 32 :
                    numPixels >= 16 ? 16 :
                    numPixels >= 8  ? 8  :
                    numPixels >= 4  ? 4  :
                    numPixels >= 2  ? 2  : 0;
        streamSettings->lacSettings[0].v3.windows.startOffset.x =
            (numRemainder + (numPixels - streamSettings->lacSettings[0].v3.windows.size.width)) / 2;

        numWindowsY = 64;
        numPixels = height / numWindowsY;
        numRemainder = height - (numPixels * numWindowsY);
        streamSettings->lacSettings[0].v3.windows.verticalInterval = numPixels;
        streamSettings->lacSettings[0].v3.windows.size.height =
                    numPixels >= 32 ? 32 :
                    numPixels >= 16 ? 16 :
                    numPixels >= 8  ? 8  :
                    numPixels >= 4  ? 4  :
                    numPixels >= 2  ? 2  : 0;
        streamSettings->lacSettings[0].v3.windows.startOffset.y =
            (numRemainder + (numPixels - streamSettings->lacSettings[0].v3.windows.size.height)) / 2;
        streamSettings->lacSettings[0].v3.windows.horizontalNum = numWindowsX;
        streamSettings->lacSettings[0].v3.windows.verticalNum = numWindowsY;
    } else {
        LOG_ERR("%s: isp version not supported\n", __func__, ctx->ispVersion);
    }
}

static void
initializeLACSettings1(
    PluginContext *ctx,
    NvMediaIPPStreamType type,
    int width,
    int height)
{
    int numWindowsX, numWindowsY;
    int numPixels;
    NvMediaIPPPluginOutputStreamSettings *streamSettings =
        &ctx->streamSettings[type];

    streamSettings->lacSettingsValid[1] = NVMEDIA_TRUE;
    numPixels = 32;

    if (ctx->ispVersion == NVMEDIA_IPP_ISP_VERSION_4) {
        int i, j, index = 0;
        int ROIOffset_x = 0, ROIOffset_y = 0;

        streamSettings->lacSettings[1].v4.enable = 1;
        streamSettings->lacSettings[1].v4.pixelFormat = NVMEDIA_ISP_PIXELFORMAT_QUAD;
        streamSettings->lacSettings[1].v4.rgbToYGain[0] = 0.29296875;
        streamSettings->lacSettings[1].v4.rgbToYGain[1] = 0.57421875;
        streamSettings->lacSettings[1].v4.rgbToYGain[2] = 0.132812545;
        streamSettings->lacSettings[1].v4.rgbToYGain[3] = 0;
        streamSettings->lacSettings[1].v4.range[0].low = 0.0;
        streamSettings->lacSettings[1].v4.range[0].high = 1.0;
        streamSettings->lacSettings[1].v4.range[1].low = 0.0;
        streamSettings->lacSettings[1].v4.range[1].high = 1.0;
        streamSettings->lacSettings[1].v4.range[2].low = 0.0;
        streamSettings->lacSettings[1].v4.range[2].high = 1.0;
        streamSettings->lacSettings[1].v4.range[3].low = 0.0;
        streamSettings->lacSettings[1].v4.range[3].high = 1.0;
        //Set up 2x2 ROI grid
        for (i=0; i<2; i++, ROIOffset_y += height/2) {
            ROIOffset_x = 0;
            for (j=0; j<2; j++, ROIOffset_x += width/2, index++) {
                streamSettings->lacSettings[1].v4.ROIEnable[index] = NVMEDIA_TRUE;

                numWindowsX = width / (numPixels * 2);
                numWindowsX = numWindowsX > 32 ? 32 : numWindowsX;

                streamSettings->lacSettings[1].v4.windows[index].horizontalInterval = width / (numWindowsX * 2);
                streamSettings->lacSettings[1].v4.windows[index].size.width = numPixels;
                streamSettings->lacSettings[1].v4.windows[index].startOffset.x =
                    (((width / 2) - streamSettings->lacSettings[1].v4.windows[index].horizontalInterval * numWindowsX) +
                    (streamSettings->lacSettings[1].v4.windows[index].horizontalInterval - numPixels)) / 2 + ROIOffset_x;

                numWindowsY = height / (numPixels * 2);
                numWindowsY = numWindowsY > 32 ? 32 : numWindowsY;

                streamSettings->lacSettings[1].v4.windows[index].verticalInterval = height / (numWindowsY * 2);
                streamSettings->lacSettings[1].v4.windows[index].size.height = numPixels;

                streamSettings->lacSettings[1].v4.windows[index].startOffset.y =
                    (((height / 2) - streamSettings->lacSettings[1].v4.windows[index].verticalInterval * numWindowsY) +
                    (streamSettings->lacSettings[1].v4.windows[index].verticalInterval - numPixels)) / 2 + ROIOffset_y;
                streamSettings->lacSettings[1].v4.windows[index].horizontalNum = numWindowsX;
                streamSettings->lacSettings[1].v4.windows[index].verticalNum = numWindowsY;
            }
        }
    } else if (ctx->ispVersion == NVMEDIA_IPP_ISP_VERSION_3) {
        streamSettings->lacSettings[1].v3.enable = 1;
        streamSettings->lacSettings[1].v3.pixelFormat = NVMEDIA_ISP_PIXELFORMAT_QUAD;
        streamSettings->lacSettings[1].v3.rgbToYGain[0] = 0.29296875;
        streamSettings->lacSettings[1].v3.rgbToYGain[1] = 0.57421875;
        streamSettings->lacSettings[1].v3.rgbToYGain[2] = 0.132812545;
        streamSettings->lacSettings[1].v3.rgbToYGain[3] = 0;
        streamSettings->lacSettings[1].v3.range[0].low = 0.0;
        streamSettings->lacSettings[1].v3.range[0].high = 1.0;
        streamSettings->lacSettings[1].v3.range[1].low = 0.0;
        streamSettings->lacSettings[1].v3.range[1].high = 1.0;
        streamSettings->lacSettings[1].v3.range[2].low = 0.0;
        streamSettings->lacSettings[1].v3.range[2].high = 1.0;
        streamSettings->lacSettings[1].v3.range[3].low = 0.0;
        streamSettings->lacSettings[1].v3.range[3].high = 1.0;

        numWindowsX = width / numPixels;
        numWindowsX = numWindowsX > 64 ? 64 : numWindowsX;
        streamSettings->lacSettings[1].v3.windows.horizontalInterval = width / numWindowsX;
        streamSettings->lacSettings[1].v3.windows.size.width = numPixels;
        streamSettings->lacSettings[1].v3.windows.startOffset.x =
            ((width - streamSettings->lacSettings[1].v3.windows.horizontalInterval * numWindowsX) +
            (streamSettings->lacSettings[1].v3.windows.horizontalInterval - numPixels)) / 2;

        numWindowsY = height / numPixels;
        numWindowsY = numWindowsY > 64 ? 64 : numWindowsY;
        streamSettings->lacSettings[1].v3.windows.verticalInterval = height / numWindowsY;
        streamSettings->lacSettings[1].v3.windows.size.height = numPixels;
        streamSettings->lacSettings[1].v3.windows.startOffset.y =
            ((height - streamSettings->lacSettings[1].v3.windows.verticalInterval * numWindowsY) +
            (streamSettings->lacSettings[1].v3.windows.verticalInterval - numPixels)) / 2;

        streamSettings->lacSettings[1].v3.windows.horizontalNum = numWindowsX;
        streamSettings->lacSettings[1].v3.windows.verticalNum = numWindowsY;
    } else {
        LOG_ERR("%s: isp version not supported\n", __func__, ctx->ispVersion);
    }
}

static void initializeFlickerbandSettings(
    PluginContext *ctx,
    NvMediaIPPStreamType type,
    int width,
    int height)
{
    int windowHeight = 64;
    int numBands = height / windowHeight;
    NvMediaIPPPluginOutputStreamSettings *streamSettings =
        &ctx->streamSettings[type];

    // For low resolution shrink the band heights to have higher precision.
    while (numBands < 64 && windowHeight > 1) {
        windowHeight /= 2;
        numBands = height / windowHeight;
    }

    // Max 256 bands
    if (numBands > 256) numBands = 256;

    streamSettings->flickerBandSettingsValid = NVMEDIA_TRUE;

    if (ctx->ispVersion == NVMEDIA_IPP_ISP_VERSION_4) {
        streamSettings->flickerBandSettings.v4.enable = NVMEDIA_TRUE;
        streamSettings->flickerBandSettings.v4.windows.size.width = width;
        streamSettings->flickerBandSettings.v4.windows.size.height = windowHeight;
        streamSettings->flickerBandSettings.v4.windows.horizontalNum = 1;
        streamSettings->flickerBandSettings.v4.windows.verticalNum = numBands;
        streamSettings->flickerBandSettings.v4.windows.horizontalInterval = 0;
        streamSettings->flickerBandSettings.v4.windows.verticalInterval = height / numBands;
        streamSettings->flickerBandSettings.v4.windows.startOffset.x = 0;
        streamSettings->flickerBandSettings.v4.windows.startOffset.y =
            (height - streamSettings->flickerBandSettings.v4.windows.verticalInterval * numBands) / 2;
        streamSettings->flickerBandSettingsValid = NVMEDIA_FALSE;
        streamSettings->flickerBandSettings.v4.windows.verticalInterval = windowHeight;
        streamSettings->flickerBandSettings.v4.colorChannel = NVMEDIA_ISP_COLORCHANNEL_TL_R_V;//LUMINANCE;
        streamSettings->flickerBandSettings.v4.hdrMode = NVMEDIA_ISP_HDR_MODE_NORMAL; //Not HDR mode
    } else if (ctx->ispVersion == NVMEDIA_IPP_ISP_VERSION_3) {
        streamSettings->flickerBandSettings.v3.enable = NVMEDIA_TRUE;
        streamSettings->flickerBandSettings.v3.windows.size.width = width;
        streamSettings->flickerBandSettings.v3.windows.size.height = windowHeight;
        streamSettings->flickerBandSettings.v3.windows.horizontalNum = 1;
        streamSettings->flickerBandSettings.v3.windows.verticalNum = numBands;
        streamSettings->flickerBandSettings.v3.windows.horizontalInterval = 0;
        streamSettings->flickerBandSettings.v3.windows.verticalInterval = height / numBands;
        streamSettings->flickerBandSettings.v3.windows.startOffset.x = 0;
        streamSettings->flickerBandSettings.v3.windows.startOffset.y =
            (height - streamSettings->flickerBandSettings.v3.windows.verticalInterval * numBands) / 2;
    } else {
        LOG_ERR("%s: isp version not supported\n", __func__, ctx->ispVersion);
    }
}

static void
initializeHistogramsettings0(
    PluginContext *ctx,
    NvMediaIPPStreamType type,
    int width,
    int height)
{
    NvMediaIPPPluginOutputStreamSettings *streamSettings =
        &ctx->streamSettings[type];

    streamSettings->histogramSettingsValid[0] = NVMEDIA_TRUE;

    if (ctx->ispVersion == NVMEDIA_IPP_ISP_VERSION_4) {
        int i;
        streamSettings->histogramSettings[0].v4.enable = NVMEDIA_TRUE;
        streamSettings->histogramSettings[0].v4.pixelFormat = NVMEDIA_ISP_PIXELFORMAT_QUAD;
        streamSettings->histogramSettings[0].v4.window.x0 = 0;
        streamSettings->histogramSettings[0].v4.window.y0 = 0;
        streamSettings->histogramSettings[0].v4.window.x1 = width;
        streamSettings->histogramSettings[0].v4.window.y1 = height;

        streamSettings->histogramSettings[0].v4.hdrMode = NVMEDIA_ISP_HDR_MODE_NORMAL; //Not HDR mode
        for (i = 0; i < NVMEDIA_ISP_HIST_RANGE_CFG_NUM; i++) {
            streamSettings->histogramSettings[0].v4.range[i] = (i<2) ? 6 : 7 + (i-2);
            streamSettings->histogramSettings[0].v4.knee[i]  = (i+1) * 32 - 1;
        }
    } else if (ctx->ispVersion == NVMEDIA_IPP_ISP_VERSION_3) {
        streamSettings->histogramSettings[0].v3.enable = NVMEDIA_TRUE;
        streamSettings->histogramSettings[0].v3.pixelFormat = NVMEDIA_ISP_PIXELFORMAT_QUAD;
        streamSettings->histogramSettings[0].v3.window.x0 = 0;
        streamSettings->histogramSettings[0].v3.window.y0 = 0;
        streamSettings->histogramSettings[0].v3.window.x1 = width;
        streamSettings->histogramSettings[0].v3.window.y1 = height;

        streamSettings->histogramSettings[0].v3.range.low = 0;
        streamSettings->histogramSettings[0].v3.range.high = 8192;
    } else {
        LOG_ERR("%s: isp version not supported\n", __func__, ctx->ispVersion);
    }
}

static void
initializeHistogramsettings1(
    PluginContext *ctx,
    NvMediaIPPStreamType type,
    int width,
    int height)
{
    NvMediaIPPPluginOutputStreamSettings *streamSettings =
        &ctx->streamSettings[type];

    streamSettings->histogramSettingsValid[1] = NVMEDIA_TRUE;

    if (ctx->ispVersion == NVMEDIA_IPP_ISP_VERSION_4) {
        int i;
        streamSettings->histogramSettings[1].v4.enable = NVMEDIA_TRUE;
        streamSettings->histogramSettings[1].v4.pixelFormat = NVMEDIA_ISP_PIXELFORMAT_RGB;
        streamSettings->histogramSettings[1].v4.window.x0 = 0;
        streamSettings->histogramSettings[1].v4.window.y0 = 0;
        streamSettings->histogramSettings[1].v4.window.x1 = width;
        streamSettings->histogramSettings[1].v4.window.y1 = height;

        streamSettings->histogramSettings[1].v4.hdrMode = NVMEDIA_ISP_HDR_MODE_NORMAL; //Not HDR mode
        for (i = 0; i < NVMEDIA_ISP_HIST_RANGE_CFG_NUM; i++) {
            streamSettings->histogramSettings[1].v4.range[i] = (i<2) ? 6 : 7 + (i-2);
            streamSettings->histogramSettings[1].v4.knee[i]  = (i+1) * 32 - 1;
        }
    } else if (ctx->ispVersion == NVMEDIA_IPP_ISP_VERSION_3) {
        streamSettings->histogramSettings[1].v3.enable = NVMEDIA_TRUE;
        streamSettings->histogramSettings[1].v3.pixelFormat = NVMEDIA_ISP_PIXELFORMAT_RGB;
        streamSettings->histogramSettings[1].v3.window.x0 = 0;
        streamSettings->histogramSettings[1].v3.window.y0 = 0;
        streamSettings->histogramSettings[1].v3.window.x1 = width;
        streamSettings->histogramSettings[1].v3.window.y1 = height;

        streamSettings->histogramSettings[1].v3.range.low = 0;
        streamSettings->histogramSettings[1].v3.range.high = 8192;
    } else {
        LOG_ERR("%s: isp version not supported\n", __func__, ctx->ispVersion);
    }
}

NvMediaStatus
IPPPluginCreate(
    NvMediaIPPComponent *controlAlgorithmHandle,
    NvMediaIPPPluginSupportFuncs *pSupportFunctions,
    NvMediaIPPPropertyStatic *pStaticProperties,
    void *clientContext,
    NvMediaIPPPlugin **pluginHandle,
    NvMediaIPPISPVersion ispVersion)
{
    PluginContext *ctx;
    NvMediaIPPExposureControl *aeControl;
    NvMediaIPPWBGainControl *awbGainControl;
    int width, height;
    NvMediaIPPPluginOutput *runPluginOutput;
    float initLongET, initLongGain,  initShortET, initShortGain, initVShortET,initVShortGain;
    unsigned int type;

    if (!pStaticProperties || !pluginHandle) {
        LOG_ERR("%s: Invalid arguemnt", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    ctx = calloc(1, sizeof(PluginContext));
    if (!ctx) {
        LOG_ERR("%s: Out of memory!\n", __func__);
        return NVMEDIA_STATUS_OUT_OF_MEMORY;
    }

    ctx->controlAlgorithmHandle = controlAlgorithmHandle;
    ctx->ispVersion = ispVersion;

    if (pSupportFunctions) {
        memcpy( &ctx->supportFunctions,
                pSupportFunctions,
                sizeof(NvMediaIPPPluginSupportFuncs));
    }

    memcpy( &ctx->staticProperties, pStaticProperties,
        sizeof(NvMediaIPPPropertyStatic));

    aeControl = &ctx->runningPluginOutput.exposureControl;
    awbGainControl = &ctx->runningPluginOutput.whiteBalanceGainControl;

    width = pStaticProperties->sensorMode.activeArraySize.width;
    height = pStaticProperties->sensorMode.activeArraySize.height;

    for(type = 0; type < NVMEDIA_IPP_STREAM_MAX_TYPES; type++) {
        initializeLACSettings0(ctx, type, width, height, NVMEDIA_ISC_SENSOR_MODE_12BIT);
        initializeLACSettings1(ctx, type, width, height);
        initializeFlickerbandSettings(ctx, type, width, height);
        initializeHistogramsettings0(ctx, type, width, height);
        initializeHistogramsettings1(ctx, type, width, height);
    }

    runPluginOutput = &ctx->runningPluginOutput;

    runPluginOutput->aeState = NVMEDIA_IPP_AE_STATE_INACTIVE;
    runPluginOutput->aeLock = NVMEDIA_FALSE;

    runPluginOutput->colorCorrectionMatrixValid = NVMEDIA_TRUE;
    runPluginOutput->colorCorrectionMatrix.array[0][0] = 1.72331000;
    runPluginOutput->colorCorrectionMatrix.array[0][1] = -0.15490000;
    runPluginOutput->colorCorrectionMatrix.array[0][2] = 0.04468000;
    runPluginOutput->colorCorrectionMatrix.array[0][3] = 0.0;

    runPluginOutput->colorCorrectionMatrix.array[1][0] = -0.64099000;
    runPluginOutput->colorCorrectionMatrix.array[1][1] = 1.46603000;
    runPluginOutput->colorCorrectionMatrix.array[1][2] = -0.78100000;
    runPluginOutput->colorCorrectionMatrix.array[1][3] = 0.0;

    runPluginOutput->colorCorrectionMatrix.array[2][0] = -0.08232000;
    runPluginOutput->colorCorrectionMatrix.array[2][1] = -0.31113000;
    runPluginOutput->colorCorrectionMatrix.array[2][2] = 1.73632000;
    runPluginOutput->colorCorrectionMatrix.array[2][3] = 0.0;

    runPluginOutput->colorCorrectionMatrix.array[3][0] = 0;
    runPluginOutput->colorCorrectionMatrix.array[3][1] = 0;
    runPluginOutput->colorCorrectionMatrix.array[3][2] = 0;
    runPluginOutput->colorCorrectionMatrix.array[3][3] = 1.0;

    initLongET = pStaticProperties->exposureTimeRange[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].high/1e9;
    initLongGain = pStaticProperties->sensorAnalogGainRange[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].high;
    initShortET = pStaticProperties->exposureTimeRange[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_SHORT].high/1e9;
    initShortGain = pStaticProperties->sensorAnalogGainRange[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_SHORT].high;
    initVShortET = pStaticProperties->exposureTimeRange[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_VERY_SHORT].high/1e9;
    initVShortGain = pStaticProperties->sensorAnalogGainRange[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_VERY_SHORT].high;

    aeControl->sensorGain[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].value = initLongGain;
    aeControl->sensorGain[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].valid = NVMEDIA_TRUE;
    aeControl->exposureTime[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].value = initLongET;
    aeControl->exposureTime[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].valid = NVMEDIA_TRUE;

    aeControl->sensorGain[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_SHORT].value = initShortGain;
    aeControl->sensorGain[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_SHORT].valid = NVMEDIA_TRUE;
    aeControl->exposureTime[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_SHORT].value = initShortET;
    aeControl->exposureTime[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_SHORT].valid = NVMEDIA_TRUE;

    aeControl->sensorGain[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_VERY_SHORT].value = initVShortGain;
    aeControl->sensorGain[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_VERY_SHORT].valid = NVMEDIA_TRUE;
    aeControl->exposureTime[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_VERY_SHORT].value = initVShortET;
    aeControl->exposureTime[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_VERY_SHORT].valid = NVMEDIA_TRUE;
    aeControl->sensorMode = NVMEDIA_ISC_SENSOR_MODE_12BIT;

    ctx->awb_Adpatgain2shortexp[0] = 0.8;
    ctx->awb_Adpatgain2shortexp[1] = 1.0;
    ctx->awb_Adpatgain2shortexp[2] = 1.0;
    ctx->awb_Adpatgain2shortexp[3] = 0.9;
    awbGainControl->wbGain[0].valid = NVMEDIA_TRUE;
    awbGainControl->wbGain[1].valid = NVMEDIA_TRUE;
    awbGainControl->wbGain[2].valid = NVMEDIA_TRUE;

    awbGainControl->wbGain[0].value[0] = 1.5314f;
    awbGainControl->wbGain[0].value[1] = 1.0f;
    awbGainControl->wbGain[0].value[2] = 1.0f;
    awbGainControl->wbGain[0].value[3] = 1.8495f;

    awbGainControl->wbGain[1].value[0] = ctx->awb_Adpatgain2shortexp[0]*1.5314f;
    awbGainControl->wbGain[1].value[1] = ctx->awb_Adpatgain2shortexp[1]*1.0f;
    awbGainControl->wbGain[1].value[2] = ctx->awb_Adpatgain2shortexp[2]*1.0f;
    awbGainControl->wbGain[1].value[3] = ctx->awb_Adpatgain2shortexp[3]*1.8495f;

    awbGainControl->wbGain[2].value[0] = 1.5314f;
    awbGainControl->wbGain[2].value[1] = 1.0f;
    awbGainControl->wbGain[2].value[2] = 1.0f;
    awbGainControl->wbGain[2].value[3] = 1.8495f;

    *pluginHandle = ctx;

    return NVMEDIA_STATUS_OK;
}

void
IPPPluginDestroy(
    NvMediaIPPPlugin *pluginHandle)
{
    if(!pluginHandle) return;

    free(pluginHandle);
}

static void
PrintPluginInput (PluginContext *ctx,
    NvMediaIPPPluginInput *pluginInput)
{
#ifdef PRINT_ISPSTATS_FORDEBUG
    NvMediaIPPImageInformation *pImageInfo = &pluginInput->imageInfo;

    NvMediaISPStatsFlickerBandMeasurement *pFlickerBandStats;
    NvMediaISPStatsHistogramMeasurement *pHistogramStats;
    NvMediaIPPPluginInputStreamData *streamData;
    unsigned int type;

    // Parsing Image Information: Printing for illustration purpose only
    // shall be used in intended plugin alogrithm
    LOG_DBG("Frame ID: %d\t & Camera ID %d\n",
                pImageInfo->frameId,
                pImageInfo->cameraId);

    LOG_DBG("FrameSeqNum %u\n", pImageInfo->frameSequenceNumber);

    // Parsing Embedded Data Information: Printing for illustration purpose only
    // shall be used in intended plugin alogrithm
    NvMediaIPPEmbeddedDataInformation *pEmbeddedDataInfo = &pluginInput->embeddedDataInfo;
    LOG_DBG("topEmbeddedDataSize: %d\t & bottomEmbeddedDataSize: %d\n",
                pEmbeddedDataInfo->topEmbeddedDataSize,
                pEmbeddedDataInfo->bottomEmbeddedDataSize);

    LOG_DBG("topBaseRegAddress: %d\t & bottomBaseRegAddress: %d\n",
                pEmbeddedDataInfo->topBaseRegAddress,
                pEmbeddedDataInfo->bottomBaseRegAddress);

    LOG_DBG("Pointer to topEmbeddedData: %p\t & bottomEmbeddedData: %p\n",
                pEmbeddedDataInfo->topEmbeddedData,
                pEmbeddedDataInfo->bottomEmbeddedData);


    // Parsing Compandig control Data associated withg currently processed image.
    // Printing for illustration purpose only,
    NvMediaIPPCompandingControl *compandControl = &pluginInput->compandingControl;
    LOG_DBG("Companding factor: %.4f\t & MaxValue: %.4f\n",
             compandControl->compandingFactor,
             compandControl->maxValue);

    LOG_DBG("HDR Ratio: %.4f\n", compandControl->hdrRatio);

    for(type = 0; type < NVMEDIA_IPP_STREAM_MAX_TYPES; type++) {
        streamData = &pluginInput->streamData[type];
        if(!streamData->enabled)
            continue;

        LOG_DBG("Statistics from stream type %u\n", type);
        if (ctx->ispVersion == NVMEDIA_IPP_ISP_VERSION_3) {
            NvMediaISPStatsLacMeasurement *pIspLacStats;
            pIspLacStats = streamData->lacStats[0].v3;
            if(pIspLacStats) {
                // LAC Stats
                LOG_DBG("[Lac Stats] Position of the top-left pixel in the top-left window: (%d, %d)\n",
                        pIspLacStats->startOffset.x,
                        pIspLacStats->startOffset.y);

                LOG_DBG("[Lac Stats] Size of each window: %dx%d \n",
                        pIspLacStats->windowSize.width,
                        pIspLacStats->windowSize.height);

                LOG_DBG("[Lac Stats] The number of horizontal windows: %d\n",pIspLacStats->numWindowsH);
                LOG_DBG("[Lac Stats] The number of vertical windows: %d\n",pIspLacStats->numWindowsV);
                LOG_DBG("[Lac Stats] Avg of Each Color Component Window #1:\n"
                        "(top_left, top_right, bottom_left, bottom_right) = "
                        "(%d, %d, %d, %d)\n",
                        pIspLacStats->average[NVMEDIA_ISP_COLOR_COMPONENT_TL][0],
                        pIspLacStats->average[NVMEDIA_ISP_COLOR_COMPONENT_TR][0],
                        pIspLacStats->average[NVMEDIA_ISP_COLOR_COMPONENT_BL][0],
                        pIspLacStats->average[NVMEDIA_ISP_COLOR_COMPONENT_BR][0]);

                LOG_DBG("[Lac Stats] Num pixels in each Color Component Window #1:\n"
                        "(top_left, top_right, bottom_left, bottom_right) = "
                        "(%d, %d, %d, %d)\n",
                        pIspLacStats->numPixels[NVMEDIA_ISP_COLOR_COMPONENT_TL][0],
                        pIspLacStats->numPixels[NVMEDIA_ISP_COLOR_COMPONENT_TR][0],
                        pIspLacStats->numPixels[NVMEDIA_ISP_COLOR_COMPONENT_BL][0],
                        pIspLacStats->numPixels[NVMEDIA_ISP_COLOR_COMPONENT_BR][0]);
            }
        } else if (ctx->ispVersion == NVMEDIA_IPP_ISP_VERSION_4) {
            NvMediaISPStatsLacMeasurementV4 *pIspLacStats;
            pIspLacStats = streamData->lacStats[0].v4;

            if(pIspLacStats) {
                int i;
                for (i=0; i<4; i++) {

                    if (0 &&pIspLacStats->ROIEnable[i]) {
                        LOG_DBG("[Lac Stats] ROI %d enabled\n", i);

                        LOG_DBG("[Lac Stats] Position of the top-left pixel in the top-left window: (%d, %d)\n",
                                pIspLacStats->startOffset[i].x,
                                pIspLacStats->startOffset[i].y);

                        LOG_DBG("[Lac Stats] Size of each window: %dx%d \n",
                                pIspLacStats->windowSize[i].width,
                                pIspLacStats->windowSize[i].height);

                        LOG_DBG("[Lac Stats] The number of horizontal windows: %d\n",pIspLacStats->numWindows[i]);
                        LOG_DBG("[Lac Stats] The number of vertical windows: %d\n",pIspLacStats->numWindowsV[i]);
                        LOG_DBG("[Lac Stats] Avg of Each Color Component Window #1:\n"
                                "(top_left, top_right, bottom_left, bottom_right) = "
                                "(%f, %f, %f, %f)\n",
                                pIspLacStats->average[i][NVMEDIA_ISP_COLOR_COMPONENT_TL][0],
                                pIspLacStats->average[i][NVMEDIA_ISP_COLOR_COMPONENT_TR][0],
                                pIspLacStats->average[i][NVMEDIA_ISP_COLOR_COMPONENT_BL][0],
                                pIspLacStats->average[i][NVMEDIA_ISP_COLOR_COMPONENT_BR][0]);

                        LOG_DBG("[Lac Stats] Num pixels in each Color Component Window #1:\n"
                                "(top_left, top_right, bottom_left, bottom_right) = "
                                "(%d, %d, %d, %d)\n",
                                pIspLacStats->numPixels[i][NVMEDIA_ISP_COLOR_COMPONENT_TL][0],
                                pIspLacStats->numPixels[i][NVMEDIA_ISP_COLOR_COMPONENT_TR][0],
                                pIspLacStats->numPixels[i][NVMEDIA_ISP_COLOR_COMPONENT_BL][0],
                                pIspLacStats->numPixels[i][NVMEDIA_ISP_COLOR_COMPONENT_BR][0]);
                    }
                }
            }
        } else {
            LOG_ERR("%s: isp version not supported\n", __func__, ctx->ispVersion);
        }

        // Histogram Stats
        pHistogramStats =  streamData->histogramStats[0];
        if(pHistogramStats) {
            LOG_DBG("[Histgrm Stats] Number of bins in each color component has: %d\n",
                        pHistogramStats->numBins);
        }

        pFlickerBandStats = streamData->flickerBandStats;
        if(pFlickerBandStats) {
            // Flicker band statistics
            LOG_DBG("[FlickerBand Stats] The number of windows: %d\n",
                    pFlickerBandStats->numWindows);

            LOG_DBG("[FlickerBand Stats] The pointer to avg Luminance: %p\n",
                    pFlickerBandStats->luminance);
        }
    }
#endif
}

static void
IPPPluginSimpleAWB(NvMediaIPPPlugin *pluginHandle,
    NvMediaIPPPluginInput *pluginInput)
{
    PluginContext *ctx = (PluginContext*)pluginHandle;
    NvMediaIPPWBGainControl *awbGainControl =
        &ctx->runningPluginOutput.whiteBalanceGainControl;

    NvMediaIPPPropertyStatic* pstaticProperties = &ctx->staticProperties;

    unsigned int j, cnt, numpixels = 0;
    float *adpatgain2shortexp = ctx->awb_Adpatgain2shortexp;
    float Gavg, Ravg, Bavg;
    float bgain, rgain, ggain;
    float prevBgain, prevRgain, prevGgain, prevGgain1, prevGgain2;
    float min, longfraction;
    float normalization, invgains[4];
    NvMediaIPPPluginInputStreamData *streamData;

    if(pluginInput->streamData[NVMEDIA_IPP_STREAM_HUMAN_VISION].enabled) {
        streamData = &pluginInput->streamData[NVMEDIA_IPP_STREAM_HUMAN_VISION];
    } else {
        streamData = &pluginInput->streamData[NVMEDIA_IPP_STREAM_MACHINE_VISION];
    }

    if(!pstaticProperties->wbGainsAppliedInISP &&
        pluginInput->whiteBalanceGainControl.wbGain[0].valid) {
        prevRgain = pluginInput->whiteBalanceGainControl.wbGain[0].value[0];
        prevGgain1 = pluginInput->whiteBalanceGainControl.wbGain[0].value[1];
        prevGgain2 = pluginInput->whiteBalanceGainControl.wbGain[0].value[2];
        prevBgain = pluginInput->whiteBalanceGainControl.wbGain[0].value[3];
    } else {
        prevRgain = 1.0f;
        prevGgain1 = 1.0f;
        prevGgain2 = 1.0f;
        prevBgain = 1.0f;
    }
    prevGgain = (prevGgain1+prevGgain2)/2;

    if(pluginInput->exposureControl.sensorMode == NVMEDIA_ISC_SENSOR_MODE_10BIT)
    {
       longfraction = (2 ^ 10) / (2 ^ 14);
    }
    else
    {
        longfraction = (2 ^ 12) / (2 ^ 14);
    }
    //ISP Lac measurement is programmed 0 to longfraction for AWB, and lac avergaes are 14-bit integers
    //Thus the lac averages are normalized to be in range 0 to 1.0, by normalization: 1/(longfraction * 2^14)
    normalization = 1.0f/(longfraction * (1<<14));//1.0f/1023;


    //Compensate for the AWB gains applied at sensor already (along with normalization, as optimization)
    invgains[0] = normalization/prevRgain;
    invgains[1] = normalization/prevGgain1;
    invgains[2] = normalization/prevGgain2;
    invgains[3] = normalization/prevBgain;

    Gavg = 0;
    Ravg = 0;
    Bavg = 0;
    cnt= 0;

    switch (ctx->ispVersion) {
        case NVMEDIA_IPP_ISP_VERSION_4:
            {
                NvMediaISPStatsLacMeasurementV4 *pIspLacStats;
                float *RAvgStats, *GAvgStats1, *GAvgStats2, *BAvgStats;
                int i;

                numpixels = 0;
                pIspLacStats = streamData->lacStats[0].v4;
                if(!pIspLacStats)
                    return;

                for(i=0; i<4; i++) {
                    RAvgStats = pIspLacStats->average[i][0];
                    GAvgStats1 = pIspLacStats->average[i][1];
                    GAvgStats2 = pIspLacStats->average[i][2];
                    BAvgStats = pIspLacStats->average[i][3];
                    numpixels += pIspLacStats->numWindows[i];
                    for(j=0; j < pIspLacStats->numWindows[i]; j++) {
                        if(RAvgStats[j] > 0.0 &&
                           GAvgStats1[j] > 0.0 &&
                           GAvgStats2[j] > 0.0 &&
                           BAvgStats[j] > 0.0) {
                            Ravg += RAvgStats[j]*invgains[0];
                            Gavg += (GAvgStats1[j]*invgains[1] + GAvgStats2[j]*invgains[2]);
                            Bavg += BAvgStats[j]*invgains[3];
                            cnt++;
                        }
                    }
                }
            }
            break;
        case NVMEDIA_IPP_ISP_VERSION_3:
        default:
            {
                NvMediaISPStatsLacMeasurement *pIspLacStats;
                int  *RAvgStats, *GAvgStats1, *GAvgStats2, *BAvgStats;

                pIspLacStats = streamData->lacStats[0].v3;
                if(!pIspLacStats)
                    return;

                RAvgStats = pIspLacStats->average[0];
                GAvgStats1 = pIspLacStats->average[1];
                GAvgStats2 = pIspLacStats->average[2];
                BAvgStats = pIspLacStats->average[3];
                numpixels = pIspLacStats->numWindowsH * pIspLacStats->numWindowsV;

                for(j=0; j < numpixels; j++) {
                    if(RAvgStats[j] > 0.0 &&
                       GAvgStats1[j] > 0.0 &&
                       GAvgStats2[j] > 0.0 &&
                       BAvgStats[j] > 0.0) {
                        Ravg += RAvgStats[j]*invgains[0];
                        Gavg += (GAvgStats1[j]*invgains[1] + GAvgStats2[j]*invgains[2]);
                        Bavg += BAvgStats[j]*invgains[3];
                        cnt++;
                    }
                }
            }
            break;
    }

    Gavg = Gavg/(2*cnt);
    Ravg = Ravg/cnt;
    Bavg = Bavg/cnt;

    LOG_DBG("AWB previously applied gains [R,G, B]:%.3f, %.3f, %.3f\n",
             prevRgain, prevGgain, prevBgain );
    LOG_DBG("AWB avgs computed: RAvg: %.3f, GAvg:%.3f, BAvg:%.3f Used Pixels :%.2f\n",
             Ravg,Gavg,Bavg, cnt*100.0f/numpixels);

    ggain = 1.0f;
    bgain = ((float)Gavg)/Bavg;
    rgain = ((float)Gavg)/Ravg;

    /* Make sure gains are not less than 1.0 */
    min = MIN(MIN(rgain, ggain), bgain);
    if (min > 0.0f){
        rgain = rgain / min;
        ggain = ggain / min;
        bgain = bgain / min;
    } else {
        rgain = prevRgain;
        ggain = prevGgain;
        bgain = prevBgain;
    }
    if(bgain > 8.0) bgain = 8.0f;
    if(rgain > 8.0) rgain = 8.0f;
    if(ggain > 8.0) ggain = 8.0f;

    if ((bgain/prevBgain < 1.02 && bgain/prevBgain > 0.98) &&
       (ggain/prevGgain < 1.02 && ggain/prevGgain > 0.98) &&
       (rgain/prevRgain < 1.02 && rgain/prevRgain > 0.98)) {
        ctx->runningPluginOutput.awbState = NVMEDIA_IPP_AWB_STATE_CONVERGED;
        LOG_DBG("AWB CONVERGED: AWB Plugin achieved to convergence State\n");
        return;
    } else {
        ctx->runningPluginOutput.awbState = NVMEDIA_IPP_AWB_STATE_SEARCHING;
    }

    awbGainControl->wbGain[0].valid = NVMEDIA_TRUE;
    awbGainControl->wbGain[1].valid = NVMEDIA_TRUE;
    awbGainControl->wbGain[2].valid = NVMEDIA_TRUE;

    awbGainControl->wbGain[0].value[0] = rgain;
    awbGainControl->wbGain[0].value[1] = ggain;
    awbGainControl->wbGain[0].value[2] = ggain;
    awbGainControl->wbGain[0].value[3] = bgain;

    awbGainControl->wbGain[1].value[0] = rgain * adpatgain2shortexp[0];
    awbGainControl->wbGain[1].value[1] = ggain * adpatgain2shortexp[1];
    awbGainControl->wbGain[1].value[2] = ggain * adpatgain2shortexp[2];
    awbGainControl->wbGain[1].value[3] = bgain * adpatgain2shortexp[3];

    /* Make sure gains are not less than 1.0 */
    min = MIN(MIN(awbGainControl->wbGain[1].value[0], awbGainControl->wbGain[1].value[1]),
              MIN(awbGainControl->wbGain[1].value[2], awbGainControl->wbGain[1].value[3]));

    awbGainControl->wbGain[1].value[0] = awbGainControl->wbGain[1].value[0] / min;
    awbGainControl->wbGain[1].value[1] = awbGainControl->wbGain[1].value[1] / min;
    awbGainControl->wbGain[1].value[2] = awbGainControl->wbGain[1].value[2] / min;
    awbGainControl->wbGain[1].value[3] = awbGainControl->wbGain[1].value[3] / min;

    awbGainControl->wbGain[2].value[0] = rgain;
    awbGainControl->wbGain[2].value[1] = ggain;
    awbGainControl->wbGain[2].value[2] = ggain;
    awbGainControl->wbGain[2].value[3] = bgain;
}

static void
IPPPluginSimpleAutoExposure(
    NvMediaIPPPlugin *pluginHandle,
    NvMediaIPPPluginInput *pluginInput)
{
    PluginContext *ctx = (PluginContext*)pluginHandle;
    NvMediaIPPPropertyStatic* pstaticProperties = &ctx->staticProperties;
    NvMediaIPPExposureControl *aeControl = &ctx->runningPluginOutput.exposureControl;
    unsigned int j;
    float CurrentLuma=0.0f;
    float maxPixVal = 16383.0f; //14bit ISP output
    //TODO: should be parsed from control properties or set based on precision
    float targetLuma = 128.0f/1024;
    float inputExptime, inputExpgain, prevExptime, prevExpgain, PrevExpVal,expVal, factAdjust;
    float curExptime, curExpgain, curSExptime, curSExpgain, curVSExptime, curVSExpgain;
    float dampingFactor, targetExpVal;
    float channelGainRatio;
    NvMediaIPPPluginInputStreamData *streamData;

    if(pluginInput->streamData[NVMEDIA_IPP_STREAM_HUMAN_VISION].enabled) {
        streamData = &pluginInput->streamData[NVMEDIA_IPP_STREAM_HUMAN_VISION];
    } else {
        streamData = &pluginInput->streamData[NVMEDIA_IPP_STREAM_MACHINE_VISION];
    }

    LOG_DBG("Current AE Gains %d %.4f %.4f  %.4f %.4f  %.4f %.4f\n",
            ctx->runningPluginOutput.aeState,
            aeControl->exposureTime[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].value,
            aeControl->sensorGain[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].value,
            aeControl->exposureTime[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_SHORT].value,
            aeControl->sensorGain[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_SHORT].value,
            aeControl->exposureTime[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_VERY_SHORT].value,
            aeControl->sensorGain[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_VERY_SHORT].value);

    aeControl->digitalGain = 1.0f;
    aeControl->hdrRatio = 64;

    {
        float minLongET = pstaticProperties->exposureTimeRange[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].low/1e9;
        float maxLongET = pstaticProperties->exposureTimeRange[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].high/1e9;
        float minLongGain = pstaticProperties->sensorAnalogGainRange[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].low;
        float maxLongGain = pstaticProperties->sensorAnalogGainRange[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].high;
        float minShortET = pstaticProperties->exposureTimeRange[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_SHORT].low/1e9;
        float maxShortET = pstaticProperties->exposureTimeRange[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_SHORT].high/1e9;
        float minShortGain = pstaticProperties->sensorAnalogGainRange[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_SHORT].low;
        float maxShortGain = pstaticProperties->sensorAnalogGainRange[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_SHORT].high;
        float minVShortET = pstaticProperties->exposureTimeRange[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_VERY_SHORT].low/1e9;
        float maxVShortET = pstaticProperties->exposureTimeRange[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_VERY_SHORT].high/1e9;
        float minVShortGain = pstaticProperties->sensorAnalogGainRange[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_VERY_SHORT].low;
        float maxVShortGain = pstaticProperties->sensorAnalogGainRange[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_VERY_SHORT].high;

        float Gavg = 0;
        float Ravg = 0;
        float Bavg = 0;
        unsigned int numpixels = 0;
        float longfraction;
        if(pluginInput->exposureControl.sensorMode == NVMEDIA_ISC_SENSOR_MODE_10BIT)
        {
           longfraction = (2 ^ 10) / (2 ^ 14);
        }
        else
        {
            longfraction = (2 ^ 12) / (2 ^ 14);
        }
        //Estimate Luminence
        switch (ctx->ispVersion) {
            case NVMEDIA_IPP_ISP_VERSION_4:
                {
                    NvMediaISPStatsLacMeasurementV4 *pIspLacStats;
                    float *RAvgStats, *GAvgStats1, *GAvgStats2, *BAvgStats;
                    int i;

                    pIspLacStats = streamData->lacStats[1].v4;
                    if(!pIspLacStats)
                       return;

                    numpixels = 0;
                    for(i=0; i<4; i++) {
                        RAvgStats = pIspLacStats->average[i][0];
                        GAvgStats1 = pIspLacStats->average[i][1];
                        GAvgStats2 = pIspLacStats->average[i][2];
                        BAvgStats = pIspLacStats->average[i][3];
                        numpixels += pIspLacStats->numWindows[i];
                        for(j=0; j < pIspLacStats->numWindows[i]; j++) {
                            Gavg += CLIP(GAvgStats1[j]/(maxPixVal*longfraction), 0, 1);
                            Gavg += CLIP(GAvgStats2[j]/(maxPixVal*longfraction), 0, 1);
                            Ravg += CLIP(RAvgStats[j]/(maxPixVal*longfraction), 0, 1);
                            Bavg += CLIP(BAvgStats[j]/(maxPixVal*longfraction), 0, 1);
                        }
                    }
                    aeControl->sensorMode = NVMEDIA_ISC_SENSOR_MODE_12BIT;
                }
                break;
            case NVMEDIA_IPP_ISP_VERSION_3:
            default:
                {
                    NvMediaISPStatsLacMeasurement *pIspLacStats;
                    int  *RAvgStats, *GAvgStats1, *GAvgStats2, *BAvgStats;

                    pIspLacStats = streamData->lacStats[1].v3;
                    if(!pIspLacStats)
                       return;

                    RAvgStats = pIspLacStats->average[0];
                    GAvgStats1 = pIspLacStats->average[1];
                    GAvgStats2 = pIspLacStats->average[2];
                    BAvgStats = pIspLacStats->average[3];
                    numpixels = pIspLacStats->numWindowsH * pIspLacStats->numWindowsV;
                    for(j=0; j < numpixels; j++) {
                        Gavg += CLIP(GAvgStats1[j]/(maxPixVal*longfraction), 0, 1);
                        Gavg += CLIP(GAvgStats2[j]/(maxPixVal*longfraction), 0, 1);
                        Ravg += CLIP(RAvgStats[j]/(maxPixVal*longfraction), 0, 1);
                        Bavg += CLIP(BAvgStats[j]/(maxPixVal*longfraction), 0, 1);
                    }
                    aeControl->sensorMode = NVMEDIA_ISC_SENSOR_MODE_12BIT;
                }
                break;
        }

        Gavg = Gavg/(2*numpixels);
        Ravg = Ravg/numpixels;
        Bavg = Bavg/numpixels;

        CurrentLuma = (Gavg + Ravg + Bavg)/3.0f;

        if((CurrentLuma - targetLuma) < 0.001f && (CurrentLuma - targetLuma) > -0.001f) {
            ctx->runningPluginOutput.aeState = NVMEDIA_IPP_AE_STATE_CONVERGED;
            LOG_DBG("Current Luma and Target Luma %.6f, %.6f, %.6f\n", CurrentLuma, targetLuma, CurrentLuma - targetLuma);
            LOG_DBG("AE CONVERGED: AE Plugin achieved to convergence State\n");
            return;
        } else {
            ctx->runningPluginOutput.aeState = NVMEDIA_IPP_AE_STATE_SEARCHING;
        }

        inputExptime = pluginInput->exposureControl.exposureTime[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].value;
        inputExpgain = pluginInput->exposureControl.sensorGain[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].value;
        prevExptime = aeControl->exposureTime[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].value;
        prevExpgain = aeControl->sensorGain[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].value;
        PrevExpVal = prevExptime * prevExpgain;
        expVal = inputExptime * inputExpgain;
        factAdjust = targetLuma/CurrentLuma;
        //Damping
        dampingFactor = 0.60;
        targetExpVal = dampingFactor*factAdjust * expVal + (1 - dampingFactor) * PrevExpVal;
        curExptime = maxLongET;
        curExpgain = targetExpVal / curExptime;
        if (curExpgain < minLongGain) {
             curExpgain = minLongGain;
             curExptime = targetExpVal / curExpgain;
             if(curExptime < minLongET)
                 curExptime = minLongET;
        } else if(curExpgain > maxLongGain) {
            curExpgain = maxLongGain;
        }

        aeControl->sensorGain[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].value = curExpgain;
        aeControl->sensorGain[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].valid = NVMEDIA_TRUE;
        aeControl->exposureTime[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].value = curExptime;
        aeControl->exposureTime[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].valid = NVMEDIA_TRUE;

        channelGainRatio = pstaticProperties->channelGainRatio;
        if(channelGainRatio == 0) {
            targetExpVal = targetExpVal/8.0; //using hardcoded hdrRatio of 8
            curSExptime = maxShortET;
            curSExpgain = targetExpVal / curSExptime;
            if (curSExpgain < minShortGain) {
                 curSExpgain = minShortGain;
                curSExptime = targetExpVal / curSExpgain;
                if(curSExptime < minShortET)
                    curSExptime = minShortET;

            } else if(curSExpgain > maxShortGain) {
                curSExpgain = maxShortGain;
            }

            targetExpVal = targetExpVal/8.0; //using hardcoded hdrRatio of 8.
            curVSExptime = maxVShortET;
            curVSExpgain = targetExpVal / curVSExptime;
            if (curVSExpgain < minVShortGain) {
                 curVSExpgain = minVShortGain;
                 curVSExptime = targetExpVal / curVSExpgain;
                if(curVSExptime < minVShortET)
                    curVSExptime = minVShortET;
            } else if(curVSExpgain > maxVShortGain) {
                curVSExpgain = maxVShortGain;
            }
        } else { //Fixed Gain Ratio Imposed by Sensor Driver requirements
            //For Short
            targetExpVal = targetExpVal/8.0; //using hardcoded hdrRatio of 8
            curSExptime = maxShortET; //Maximum possible
            curSExpgain = curExpgain/channelGainRatio; //Minimum possible
            if((curSExpgain * curSExptime) > targetExpVal) {
                //Reduce Exp time, as  min gain is still higher
                curSExptime = targetExpVal / curSExpgain;
                 if(curSExptime < minShortET)
                    curSExptime = minShortET;
            } else {
                curSExpgain = curExpgain;
                if((curSExpgain * curSExptime) > targetExpVal) {
                    curSExptime = targetExpVal / curSExpgain;
                    if(curSExptime < minShortET)
                       curSExptime = minShortET;
                } else {
                    curSExpgain = curExpgain * channelGainRatio; //Maximum possible
                    curSExptime = targetExpVal / curSExpgain;
                    if(curSExptime < minShortET)
                       curSExptime = minShortET;
                }
            }

            if(curSExpgain < minShortGain)
            {
                curSExpgain = minShortGain;
                curSExptime = targetExpVal / curSExpgain;
                if(curSExptime > maxShortET)
                   curSExptime = maxShortET;
            }
            //For Very Short
            targetExpVal = targetExpVal/8.0; //using hardcoded hdrRatio of 8.
            curVSExptime = maxVShortET;
            curVSExpgain = curSExpgain/channelGainRatio; //Minimum possible
            if((curVSExpgain * curVSExptime) > targetExpVal) {
                //Reduce Exp time, as  min gain is still higher
                curVSExptime = targetExpVal / curVSExpgain;
                 if(curVSExptime < minVShortET)
                    curVSExptime = minVShortET;
            } else {
                curVSExpgain = curSExpgain;
                if((curVSExpgain * curVSExptime) > targetExpVal) {
                    curVSExptime = targetExpVal / curVSExpgain;
                    if(curVSExptime < minVShortET)
                       curVSExptime = minVShortET;
                } else {
                    curVSExpgain = curSExpgain*channelGainRatio; //Maximum possible
                    curVSExptime = targetExpVal / curVSExpgain;
                    if(curVSExptime < minVShortET)
                       curVSExptime = minVShortET;
                }
            }
            if(curVSExpgain < minVShortGain)
            {
                curVSExpgain = minVShortGain;
                curVSExptime = targetExpVal / curVSExpgain;
                if(curVSExptime > maxVShortET)
                   curVSExptime = maxVShortET;
            }
        }

        aeControl->sensorGain[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_SHORT].value = curSExpgain;
        aeControl->sensorGain[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_SHORT].valid = NVMEDIA_TRUE;
        aeControl->exposureTime[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_SHORT].value = curSExptime;
        aeControl->exposureTime[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_SHORT].valid = NVMEDIA_TRUE;

        aeControl->sensorGain[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_VERY_SHORT].value = curVSExpgain;
        aeControl->sensorGain[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_VERY_SHORT].valid = NVMEDIA_TRUE;
        aeControl->exposureTime[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_VERY_SHORT].value = curVSExptime;
        aeControl->exposureTime[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_VERY_SHORT].valid = NVMEDIA_TRUE;


        LOG_DBG("AE gains Estimated %.6f %.6f %.6f %.6f %.6f %.6f\n",
            curExpgain,
            curExptime,
            curSExpgain,
            curSExptime,
            curVSExpgain,
            curVSExptime);
    }
}

NvMediaStatus
IPPPluginProcess(
    NvMediaIPPPlugin *pluginHandle,
    NvMediaIPPPluginInput *pluginInput,
    NvMediaIPPPluginOutput *pluginOutput)
{
    PluginContext *ctx = (PluginContext *) pluginHandle;
    NvMediaIPPPluginOutput *runPluginOutput = &ctx->runningPluginOutput;
    if(pluginInput) {
        // Parsing Image Information: Printing for illustration purpose only
        PrintPluginInput(ctx, pluginInput);
    }

    runPluginOutput->lensShadingControlValid = NVMEDIA_FALSE;

    if(pluginInput) {
        if(runPluginOutput->aeState == NVMEDIA_IPP_AE_STATE_INACTIVE )
        {
            runPluginOutput->aeState = NVMEDIA_IPP_AE_STATE_SEARCHING;
        }

        if(pluginInput->controlsProperties->aeLock == NVMEDIA_FALSE &&
            pluginInput->controlsProperties->aeMode == NVMEDIA_IPP_AE_MODE_ON) {
            IPPPluginSimpleAutoExposure(pluginHandle, pluginInput);
        }
        else if(pluginInput->controlsProperties->aeLock == NVMEDIA_TRUE) {
            //invalidate exposure settings as per request: Bug# 1625741
            NvMediaIPPExposureControl *aeControl = &ctx->runningPluginOutput.exposureControl;
            aeControl->exposureTime[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].valid = NVMEDIA_FALSE;
            aeControl->sensorGain[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_LONG].valid = NVMEDIA_FALSE;
            aeControl->exposureTime[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_SHORT].valid = NVMEDIA_FALSE;
            aeControl->sensorGain[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_SHORT].valid = NVMEDIA_FALSE;
            aeControl->exposureTime[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_VERY_SHORT].valid = NVMEDIA_FALSE;
            aeControl->sensorGain[NVMEDIA_IPP_SENSOR_EXPOSURE_MODE_VERY_SHORT].valid = NVMEDIA_FALSE;
        }
        else if(pluginInput->controlsProperties->aeMode == NVMEDIA_IPP_AE_MODE_OFF) {
            //copy manual exposure values
            memcpy(&ctx->runningPluginOutput.exposureControl,
                &pluginInput->controlsProperties->exposureControl, sizeof(NvMediaIPPExposureControl));
        }

        if(ctx->runningPluginOutput.awbState == NVMEDIA_IPP_AWB_STATE_INACTIVE ) {
            ctx->runningPluginOutput.awbState = NVMEDIA_IPP_AWB_STATE_SEARCHING;
        }
        if(pluginInput->controlsProperties->awbLock == NVMEDIA_FALSE &&
            pluginInput->controlsProperties->awbMode == NVMEDIA_IPP_AWB_MODE_ON) {
            IPPPluginSimpleAWB(pluginHandle, pluginInput);
        }
        else if(pluginInput->controlsProperties->awbLock == NVMEDIA_TRUE) {
            //invalidate awb settings as per request: Bug# 1625741
            NvMediaIPPWBGainControl *awbGainControl = &ctx->runningPluginOutput.whiteBalanceGainControl;
            awbGainControl->wbGain[0].valid = NVMEDIA_FALSE;
            awbGainControl->wbGain[1].valid = NVMEDIA_FALSE;
            awbGainControl->wbGain[2].valid = NVMEDIA_FALSE;
        }
        else if(pluginInput->controlsProperties->awbMode == NVMEDIA_IPP_AWB_MODE_OFF) {
            //copy WB gain values
            memcpy(&ctx->runningPluginOutput.whiteBalanceGainControl,
                &pluginInput->controlsProperties->wbGains, sizeof(NvMediaIPPWBGainControl));
        }

        ctx->runningPluginOutput.aeLock = pluginInput->controlsProperties->aeLock;
        ctx->runningPluginOutput.awbLock = pluginInput->controlsProperties->awbLock;
    }

    memcpy( runPluginOutput->streamSettings,
            ctx->streamSettings,
            sizeof(runPluginOutput->streamSettings));
    LOG_DBG("Long AWB gains computed with Grayworld Alg %.3f, %.3f %.3f, %.3f\n",
            runPluginOutput->whiteBalanceGainControl.wbGain[0].value[0],
            runPluginOutput->whiteBalanceGainControl.wbGain[0].value[1],
            runPluginOutput->whiteBalanceGainControl.wbGain[0].value[2],
            runPluginOutput->whiteBalanceGainControl.wbGain[0].value[3]);

    LOG_DBG("Short AWB gains computed with Grayworld Alg %.3f, %.3f %.3f, %.3f\n",
            runPluginOutput->whiteBalanceGainControl.wbGain[1].value[0],
            runPluginOutput->whiteBalanceGainControl.wbGain[1].value[1],
            runPluginOutput->whiteBalanceGainControl.wbGain[1].value[2],
            runPluginOutput->whiteBalanceGainControl.wbGain[1].value[3]);

    //copy running plugin outputs into pluginoutputs to apply
    memcpy(pluginOutput, runPluginOutput, sizeof(NvMediaIPPPluginOutput));

   return NVMEDIA_STATUS_OK;
}

#endif // USE_CSI_OV10640
