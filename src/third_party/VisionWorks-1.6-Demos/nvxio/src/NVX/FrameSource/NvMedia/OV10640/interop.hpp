/*
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef NVMEDIA_INTEROP_HPP
#define NVMEDIA_INTEROP_HPP

#ifdef USE_CSI_OV10640

#include <nvcommon.h>
#include <nvmedia.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "main.hpp"
#include "eglstrm_setup.hpp"
#include "log_utils.h"
#include "ipp_raw.hpp"

typedef struct {

    // EGLStreams
    NvMediaDevice              *device;
    // Win Util
    EGLDisplay                  eglDisplay;
    EglStreamClient            *eglStrmCtx;
    void                       *producerCtx;
    void                       *consumerCtx;
    NvThread                   *getOutputThread;

    //EGLStreams Params
    NvMediaSurfaceType          eglProdSurfaceType;
    NvMediaBool                 producerExited;
    NvMediaBool                 consumerExited;
    NvMediaBool                 consumerInitDone;
    NvMediaBool                 interopExited;
    NvMediaBool                *quit;

    // General processing params
    NvU32                       width;
    NvU32                       height;
    NvU32                       ippNum;
    NvMediaIPPComponent        *outputComponent[NVMEDIA_MAX_AGGREGATE_IMAGES];
    NvMediaBool                 showTimeStamp;
    NvMediaBool                 showMetadataFlag;

} InteropContext;

/*  IPPInteropInit: Initiliaze context for  Interop
    CUDA consumer, NvMedia producer*/
NvMediaStatus
InteropInit (InteropContext  *interopCtx, IPPCtx *ippCtx);

/*  IPPInteropProc: Starts Interop process
    CUDA consumer, NvMedia producer*/
NvMediaStatus
InteropProc (void* data);

/*  IPPInteropFini: clears context for  Interop
    CUDA consumer, NvMedia producer*/
NvMediaStatus
InteropFini(InteropContext  *interopCtx);

#endif // USE_CSI_OV10640

#endif // NVMEDIA_INTEROP_HPP
