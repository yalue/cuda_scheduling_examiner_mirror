/*
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef NVMEDIA_EGLSTRM_SETUP_HPP
#define NVMEDIA_EGLSTRM_SETUP_HPP

#ifdef USE_CSI_OV10640

#include "nvmedia_eglstream.h"
#include "nvcommon.h"

#include "FrameSource/EGLAPIAccessors.hpp"

/* struct to give client params of the connection */
/* struct members are read-only to client */
typedef struct _EglStreamClient {
    EGLDisplay   display;
    EGLStreamKHR eglStream[NVMEDIA_MAX_AGGREGATE_IMAGES];
    NvBool       fifoMode;
    NvU32        numofStream;
} EglStreamClient;

EglStreamClient*
EGLStreamInit(EGLDisplay display,
                        NvU32 numOfStreams,
                        NvBool fifoMode);
NvMediaStatus
EGLStreamFini(EglStreamClient *client);

#endif // USE_CSI_OV10640

#endif // NVMEDIA_EGLSTRM_SETUP_HPP
