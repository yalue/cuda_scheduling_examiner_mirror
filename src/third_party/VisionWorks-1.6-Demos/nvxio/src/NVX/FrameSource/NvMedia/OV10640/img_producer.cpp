/* Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifdef USE_CSI_OV10640

#include "img_producer.hpp"
#include "buffer_utils.h"
#include "eglstrm_setup.hpp"

using namespace nvidiaio::egl_api;

// Print metadata information
static void PrintMetadataInfo(
    NvMediaIPPComponent *outputComponet,
    NvMediaIPPComponentOutput *output);

static void PrintMetadataInfo(
    NvMediaIPPComponent *outputComponet,
    NvMediaIPPComponentOutput *output)
{
    NvMediaIPPImageInformation imageInfo;
    NvMediaIPPPropertyControls control;
    NvMediaIPPPropertyDynamic dynamic;
    NvMediaIPPEmbeddedDataInformation embeddedDataInfo;
    NvU32 topSize, bottomSize;

    if(!output || !output->metadata) {
        return;
    }

    NvMediaIPPMetadataGet(
        output->metadata,
        NVMEDIA_IPP_METADATA_IMAGE_INFO,
        &imageInfo,
        sizeof(imageInfo));

    LOG_DBG("Metadata %p: frameId %u, frame sequence #%u, cameraId %u\n",
        output->metadata, imageInfo.frameId,
        imageInfo.frameSequenceNumber, imageInfo.cameraId);

    NvMediaIPPMetadataGet(
        output->metadata,
        NVMEDIA_IPP_METADATA_CONTROL_PROPERTIES,
        &control,
        sizeof(control));

    NvMediaIPPMetadataGet(
        output->metadata,
        NVMEDIA_IPP_METADATA_DYNAMIC_PROPERTIES,
        &dynamic,
        sizeof(dynamic));

    NvMediaIPPMetadataGet(
        output->metadata,
        NVMEDIA_IPP_METADATA_EMBEDDED_DATA_INFO,
        &embeddedDataInfo,
        sizeof(embeddedDataInfo));

    LOG_DBG("Metadata %p: embedded data top (base, size) = (%#x, %u)\n",
        output->metadata,
        embeddedDataInfo.topBaseRegAddress,
        embeddedDataInfo.topEmbeddedDataSize);

    LOG_DBG("Metadata %p: embedded data bottom (base, size) = (%#x, %u)\n",
        output->metadata,
        embeddedDataInfo.bottomBaseRegAddress,
        embeddedDataInfo.bottomEmbeddedDataSize);

    topSize = NvMediaIPPMetadataGetSize(
        output->metadata,
        NVMEDIA_IPP_METADATA_EMBEDDED_DATA_TOP);

    bottomSize = NvMediaIPPMetadataGetSize(
        output->metadata,
        NVMEDIA_IPP_METADATA_EMBEDDED_DATA_BOTTOM);
    if( topSize != embeddedDataInfo.topEmbeddedDataSize ||
        bottomSize != embeddedDataInfo.bottomEmbeddedDataSize ) {
        LOG_ERR("Metadata %p: embedded data sizes mismatch\n",
            output->metadata);
    }
}

// RAW Pipeline
static NvMediaStatus
SendIPPHumanVisionOutToEglStream(
    ImageProducerCtx *ctx,
    NvU32 ippNum,
    NvMediaIPPComponentOutput *output)
{
    NvMediaImage *retImage = NULL;
    NvU32 timeoutMS = EGL_PRODUCER_TIMEOUT_MS * ctx->ippNum;
    int retry = EGL_PRODUCER_GET_IMAGE_MAX_RETRIES;
    NvMediaIPPComponentOutput retOutput;
    void *ltmData;
    NvU32 ltmDataSize = 0;
    ImageProducerCtx*   eglStrmProducerCtx;
    NvMediaStatus status = NvMediaIPPMetadataGetAddress(output->metadata,
                                          NVMEDIA_IPP_METADATA_LTM_DATA,
                                          &ltmData,
                                          &ltmDataSize);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("Error getting LTM info from metadata\n");
        return status;
    }

    eglStrmProducerCtx = ctx;

    // Send Meta Data before post image
    // Send Meta Data Size
    status = NvMediaEglStreamProducerPostMetaData(eglStrmProducerCtx->eglProducer[ippNum],          // producer
                                                  0,                                 // block id
                                                  &ltmDataSize,                      // dataBuf
                                                  0,                                 // offset
                                                  4);                                // size
    if (status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaEglStreamProducerPostMetaData \
                 send Metadata size failed %d", __func__, ippNum);
        *ctx->quit = NVMEDIA_TRUE;
        return status;
    }

    // Send Meta Data
    status = NvMediaEglStreamProducerPostMetaData(eglStrmProducerCtx->eglProducer[ippNum], // producer
                                                  0,                        // block id
                                                  ltmData,                  // dataBuf
                                                  4,                        // offset
                                                  ltmDataSize);             // size
    if (status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaEglStreamProducerPostMetaData \
                 send Metadata failed %d", __func__, ippNum);
        *ctx->quit = NVMEDIA_TRUE;
        return status;
    }

    LOG_DBG("%s: EGL producer: Post image %p\n", __func__, output->image);
    if(IsFailed(NvMediaEglStreamProducerPostImage(eglStrmProducerCtx->eglProducer[ippNum],
                                                  output->image,
                                                  NULL))) {
        LOG_ERR("%s: NvMediaEglStreamProducerPostImage failed\n", __func__);
        return status;
    }

    // The first ProducerGetImage() has to happen
    // after the second ProducerPostImage()
    if(!ctx->eglProducerGetImageFlag[ippNum]) {
        ctx->eglProducerGetImageFlag[ippNum] = NVMEDIA_TRUE;
        return status;
    }

    // get image from eglstream and release it
    do {
        status = NvMediaEglStreamProducerGetImage(eglStrmProducerCtx->eglProducer[ippNum],
                                                  &retImage,
                                                  timeoutMS);
        retry--;
    } while(retry >= 0 && !retImage && !(*(ctx->quit)));

    if(retImage && status == NVMEDIA_STATUS_OK) {
        LOG_DBG("%s: EGL producer # %d: Got image %p\n", __func__, ippNum, retImage);
        retOutput.image = retImage;
        // Return processed image to IPP
        status = NvMediaIPPComponentReturnOutput(ctx->outputComponent[ippNum], //component
                                                 &retOutput);                //output image
        if (status != NVMEDIA_STATUS_OK) {
            LOG_ERR("%s: NvMediaIPPComponentReturnOutput failed %d", __func__, ippNum);
            *ctx->quit = NVMEDIA_TRUE;
            return status;
        }
    }
    else {
        LOG_DBG("%s: EGL producer: no return image\n", __func__);
        *ctx->quit = NVMEDIA_TRUE;
        status = NVMEDIA_STATUS_ERROR;
    }
    return status;
}

void
ImageProducerProc (
    void *data,
    void *user_data)
{
    ImageProducerCtx *ctx = (ImageProducerCtx *)data;
    NvMediaStatus status;
    NvU32 i;
    if(!ctx) {
        LOG_ERR("%s: Bad parameter\n", __func__);
        return;
    }

    LOG_INFO("Get IPP output thread is active, ippNum=%d\n", ctx->ippNum);
    while(!(*ctx->quit)) {
        for(i = 0; i < ctx->ippNum; i++) {
            NvMediaIPPComponentOutput output;

            // Get images from ipps
            status = NvMediaIPPComponentGetOutput(ctx->outputComponent[i], //component
                                                  GET_FRAME_TIMEOUT,       //millisecondTimeout,
                                                  &output);                //output image

            if (status == NVMEDIA_STATUS_OK) {
                if(ctx->showTimeStamp) {
                    NvMediaGlobalTime globalTimeStamp;

                    if(IsSucceed(NvMediaImageGetGlobalTimeStamp(output.image, &globalTimeStamp))) {
                        LOG_INFO("IPP: Pipeline: %d Timestamp: %lld.%06lld\n", i,
                            globalTimeStamp / 1000000, globalTimeStamp % 1000000);
                    } else {
                        LOG_ERR("%s: Get time-stamp failed\n", __func__);
                        *ctx->quit = NVMEDIA_TRUE;
                    }
                }

                if(ctx->showMetadataFlag) {
                    PrintMetadataInfo(ctx->outputComponent[i], &output);
                }

                status = SendIPPHumanVisionOutToEglStream(ctx,i,&output);

                if(status != NVMEDIA_STATUS_OK) {
                    *ctx->quit = NVMEDIA_TRUE;
                    break;
                }
            }
        } // for loop
    } // while loop

    *ctx->producerExited = NVMEDIA_TRUE;
}

ImageProducerCtx*
ImageProducerInit(NvMediaDevice *device,
                  EglStreamClient *streamClient,
                  NvU32 width, NvU32 height,
                  InteropContext *interopCtx)
{
    NvU32 i;
    ImageProducerCtx *client = NULL;

    if(!device) {
        LOG_ERR("%s: invalid NvMedia device\n", __func__);
        return NULL;
    }

    client = (ImageProducerCtx *)malloc(sizeof(ImageProducerCtx));
    if (!client) {
        LOG_ERR("%s:: failed to alloc memory\n", __func__);
        return NULL;
    }
    memset(client, 0, sizeof(ImageProducerCtx));

    client->device = device;
    client->width = width;
    client->height = height;
    client->ippNum = interopCtx->ippNum;
    client->surfaceType = interopCtx->eglProdSurfaceType;
    client->eglDisplay = streamClient->display;
    client->producerExited = &interopCtx->producerExited;
    client->quit = interopCtx->quit;
    client->showTimeStamp = interopCtx->showTimeStamp;
    client->showMetadataFlag = interopCtx->showMetadataFlag;

    for(i=0; i< interopCtx->ippNum; i++) {
        client->outputComponent[i] = interopCtx->outputComponent[i];
        // Create EGL stream producer
        EGLint streamState = 0;
        client->eglStream[i]   = streamClient->eglStream[i];
        while(streamState != EGL_STREAM_STATE_CONNECTING_KHR) {
           if(!eglQueryStreamKHR(streamClient->display,
                                 streamClient->eglStream[i],
                                 EGL_STREAM_STATE_KHR,
                                 &streamState)) {
               LOG_ERR("eglQueryStreamKHR EGL_STREAM_STATE_KHR failed\n");
            }
        }

        client->eglProducer[i] = NvMediaEglStreamProducerCreate(client->device,
                                                                client->eglDisplay,
                                                                client->eglStream[i],
                                                                client->surfaceType,
                                                                client->width/client->ippNum,
                                                                client->height);
        if(!client->eglProducer[i]) {
            LOG_ERR("%s: Failed to create EGL producer\n", __func__);
            goto fail;
        }
    }
    return client;
fail:
    ImageProducerFini(client);
    return NULL;
}

NvMediaStatus ImageProducerFini(ImageProducerCtx *ctx)
{
    NvU32 i;
    NvMediaImage *retImage = NULL;
    NvMediaIPPComponentOutput output;
    LOG_DBG("ImageProducerFini: start\n");
    if(ctx) {
        for(i = 0; i < ctx->ippNum; i++) {
            // Finalize
            do {
                retImage = NULL;
                NvMediaEglStreamProducerGetImage(ctx->eglProducer[i],
                                                 &retImage,
                                                 0);
                if(retImage) {
                    LOG_DBG("%s: EGL producer: Got image %p\n", __func__, retImage);
                    output.image = retImage;
                    NvMediaIPPComponentReturnOutput(ctx->outputComponent[i], //component
                                                        &output);                //output image
                }
            } while(retImage);
        }

        for(i=0; i<ctx->ippNum; i++) {
            if(ctx->eglProducer[i])
                NvMediaEglStreamProducerDestroy(ctx->eglProducer[i]);
        }
        free(ctx);
    }
    LOG_DBG("ImageProducerFini: end\n");
    return NVMEDIA_STATUS_OK;
}

#endif // USE_CSI_OV10640
