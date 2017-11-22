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

#include <stdlib.h>
#include <string.h>

#include "ipp_raw.hpp"
#include "eglstrm_setup.hpp"

using namespace nvidiaio::egl_api;

static EGLStreamKHR
EGLStreamCreate(EGLDisplay display,
                NvBool fifoMode)
{
#ifdef EGL_NV_stream_metadata

    static const EGLint streamAttrMailboxMode[] = {
        EGL_METADATA0_SIZE_NV, 32*1024,
        EGL_METADATA1_SIZE_NV, 16*1024,
        EGL_METADATA2_SIZE_NV, 16*1024,
        EGL_METADATA3_SIZE_NV, 16*1024, EGL_NONE };

#else

    static const EGLint streamAttrMailboxMode[] = { EGL_NONE };

#endif // EGL_NV_stream_metadata

    static const EGLint streamAttrFIFOMode[] = {
        EGL_STREAM_FIFO_LENGTH_KHR, 4,
        EGL_METADATA0_SIZE_NV, 32*1024,
        EGL_METADATA1_SIZE_NV, 16*1024,
        EGL_METADATA2_SIZE_NV, 16*1024,
        EGL_METADATA3_SIZE_NV, 16*1024, EGL_NONE };

    return eglCreateStreamKHR(display,
                              fifoMode ? streamAttrFIFOMode : streamAttrMailboxMode);
}

EglStreamClient*
EGLStreamInit(EGLDisplay display,
                        NvU32 numOfStreams,
                        NvBool fifoMode) {
    NvU32 i;
    EglStreamClient *client = NULL;

    client = (EglStreamClient *)malloc(sizeof(EglStreamClient));
    if (!client) {
        LOG_ERR("%s:: failed to alloc memory\n", __func__);
        return NULL;
    }

    client->numofStream = numOfStreams;
    client->display = display;
    client->fifoMode = fifoMode;

    for(i=0; i< numOfStreams; i++) {
        // Create with FIFO mode
        client->eglStream[i] = EGLStreamCreate(display, client->fifoMode);

        if(!eglStreamAttribKHR(client->display, client->eglStream[i], EGL_CONSUMER_LATENCY_USEC_KHR, 16000)) {
            LOG_ERR("EGLStreamSetAttr: eglStreamAttribKHR EGL_CONSUMER_LATENCY_USEC_KHR failed\n");
            goto fail;
        }
        if(!eglStreamAttribKHR(client->display, client->eglStream[i], EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR, 16000)) {
            LOG_ERR("EGLStreamSetAttr: eglStreamAttribKHR EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR failed\n");
            goto fail;
        }
    }
    return client;
fail:
    EGLStreamFini(client);
    return NULL;
}

NvMediaStatus EGLStreamFini(EglStreamClient *client) {
    NvU32 i;
    if(client) {
        for(i=0; i<client->numofStream; i++) {
            if(client->eglStream[i])
                eglDestroyStreamKHR(client->display, client->eglStream[i]);
        }
        free(client);
    }
    return NVMEDIA_STATUS_OK;
}

#endif // USE_CSI_OV10640
