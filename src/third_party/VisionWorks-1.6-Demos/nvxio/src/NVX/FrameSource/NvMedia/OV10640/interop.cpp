/*
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifdef USE_CSI_OV10640

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>

#include "log_utils.h"
#include "interop.hpp"
#include "img_producer.hpp"

#include <cuda_runtime.h>

NvMediaStatus
InteropInit (
    InteropContext  *interopCtx,
    IPPCtx *ippCtx)
{
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    NvU32 i, fifoMode;
    if (!interopCtx || !ippCtx) {
        LOG_ERR("%s: Bad parameter", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /*Initializing Interop Params*/
    interopCtx->ippNum = ippCtx->imagesNum;
    interopCtx->width  = ippCtx->inputWidth / ippCtx->imagesNum;
    interopCtx->height = ippCtx->inputHeight;
    interopCtx->device = ippCtx->device;
    interopCtx->quit   = &ippCtx->quit;
    interopCtx->showTimeStamp = ippCtx->showTimeStamp;
    interopCtx->showMetadataFlag = ippCtx->showMetadataFlag;

    /* Create egl Util, producer, consumer*/
    fifoMode = 0;

    if (!nvidiaio::egl_api::setupEGLExtensions())
    {
        LOG_ERR("%s: failed to initialize egl api \n");
        return NVMEDIA_STATUS_ERROR;
    }

    interopCtx->eglDisplay = nvidiaio::EGLDisplayAccessor::getInstance();
    if(interopCtx->eglDisplay == EGL_NO_DISPLAY)
    {
        LOG_ERR("%s: failed to initialize egl \n");
        return NVMEDIA_STATUS_ERROR;
    }

    // Stream Init
    interopCtx->eglStrmCtx = EGLStreamInit(interopCtx->eglDisplay,
                        interopCtx->ippNum, fifoMode);
    if(!interopCtx->eglStrmCtx) {
        LOG_ERR("%s: failed to create egl stream ctx \n");
        status = NVMEDIA_STATUS_ERROR;
        goto failed;
    }

    interopCtx->eglProdSurfaceType = NvMediaSurfaceType_Image_YUV_420;
    for(i = 0; i < interopCtx->ippNum; i++) {
        interopCtx->outputComponent[i] = ippCtx->outputComponent[i];
    }

    // INTEROP
    interopCtx->consumerCtx = ippCtx->cudaConnection;
    ippCtx->eglDisplay = interopCtx->eglStrmCtx->display;

    for (i = 0; i < interopCtx->ippNum; ++i)
        ippCtx->eglStream[i] = interopCtx->eglStrmCtx->eglStream[i];

    return NVMEDIA_STATUS_OK;

failed:
    LOG_ERR("%s: Failed", __func__);
    InteropFini(interopCtx);
    return (status);
}

NvMediaStatus InteropProc (void* data)
{
    InteropContext  *interopCtx = NULL;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    if(!data) {
        LOG_ERR("%s: Bad parameter\n", __func__);
        return status;
    }
    interopCtx = (InteropContext *)data;
    interopCtx->consumerInitDone = NVMEDIA_FALSE;
    interopCtx->consumerExited = NVMEDIA_FALSE;

    // Initialize EGL stream consumer
    {
        if (cudaSuccess != cudaFree(NULL))
            goto failed;

        // Create EGL stream consumer
        for (NvU32 i = 0; i < interopCtx->ippNum; ++i)
        {
            CUeglStreamConnection * connection = (CUeglStreamConnection *)interopCtx->consumerCtx + i;
            CUresult curesult = cuEGLStreamConsumerConnect(connection, interopCtx->eglStrmCtx->eglStream[i]);

            if (CUDA_SUCCESS != curesult)
            {
                printf("Connect CUDA EGL stream consumer ERROR %d\n", curesult);
                goto failed;
            }
        }

        interopCtx->consumerInitDone = NVMEDIA_TRUE;
    }

    interopCtx->producerExited = NVMEDIA_FALSE;
    interopCtx->eglProdSurfaceType = NvMediaSurfaceType_Image_YUV_420;
    interopCtx->producerCtx = ImageProducerInit(interopCtx->device,
                                                interopCtx->eglStrmCtx,
                                                interopCtx->width,
                                                interopCtx->height,
                                                interopCtx);
    if(!interopCtx->producerCtx)
    {
        LOG_ERR("%s: Failed to Init Image Producer", __func__);
        goto failed;
    }

    while (!(*interopCtx->quit) && !(interopCtx->consumerInitDone)) {
        usleep(1000);
        LOG_DBG("Waiting for consumer init to happen\n");
    }

    typedef NvU32 (*pFunc)(void *pParam);

    if(IsFailed(NvThreadCreate(&interopCtx->getOutputThread,
                               (pFunc)&ImageProducerProc,
                               (void *)interopCtx->producerCtx,
                               NV_THREAD_PRIORITY_NORMAL))) {
        interopCtx->producerExited = NVMEDIA_TRUE;
        goto failed;
    }

    return NVMEDIA_STATUS_OK;
failed:
    LOG_ERR("%s: InteropProc Failed", __func__);
    interopCtx->producerExited = NVMEDIA_TRUE;
    interopCtx->consumerExited = NVMEDIA_TRUE;
    return status;

}
NvMediaStatus
InteropFini (
    InteropContext  *interopCtx)
{
    if (!interopCtx)
        return NVMEDIA_STATUS_OK;

    while (!interopCtx->producerExited)
    {
        LOG_DBG("%s: Waiting for producer thread to quit\n", __func__);
        usleep(100);
    }

    // Image Producer Fini
    if(IsFailed(ImageProducerFini((ImageProducerCtx *)interopCtx->producerCtx))) {
        LOG_ERR("%s: ImageProducerFini failed \n", __func__);
    }

    // Finalize EGL stream consumer
    interopCtx->consumerExited = NVMEDIA_TRUE;

    // initialize CUDA context
    if (cudaSuccess == cudaFree(NULL))
    {
        for (NvU32 i = 0; i < interopCtx->ippNum; ++i)
        {
            CUeglStreamConnection * connection = (CUeglStreamConnection *)interopCtx->consumerCtx + i;
            CUresult curesult = cuEGLStreamConsumerDisconnect(connection);

            if (CUDA_SUCCESS != curesult)
                printf("Disconnect CUDA EGL stream consumer ERROR %d\n", curesult);
        }
    }

    // Stream Fini
    if(IsFailed(EGLStreamFini(interopCtx->eglStrmCtx))) {
        LOG_ERR("%s: EGLStreamFini failed \n", __func__);
    }

    if (interopCtx->getOutputThread)
        NvThreadDestroy(interopCtx->getOutputThread);

    return NVMEDIA_STATUS_OK;
}

#endif // USE_CSI_OV10640
