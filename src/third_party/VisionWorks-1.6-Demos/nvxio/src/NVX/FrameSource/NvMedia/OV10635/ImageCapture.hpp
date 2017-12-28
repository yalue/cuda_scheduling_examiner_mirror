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

#ifndef IMGCAPTURE_H
#define IMGCAPTURE_H

#ifdef USE_CSI_OV10635

#include "board_name.h"

#include "image_capture.h"
#include "image_surf_utils.h"

#include <nvmedia_isc.h>

#if BOARD_CODE_NAME == JETSON_PRO_BOARD
# include "config_isc.h"
#else
# include "img_dev.h"
#endif

#include <string>

#include <nvmedia.h>
#include <nvmedia_eglstream.h>
#include <cudaEGL.h>

#include "FrameSource/EGLAPIAccessors.hpp"

#define QUEUE_ENQUEUE_TIMEOUT           100
#define QUEUE_DEQUEUE_TIMEOUT           100

#include "FrameSource/NvMedia/NvMediaCameraConfigParams.hpp"

namespace nvidiaio { namespace ov10635 {

typedef struct {
    NvThread                   *thread;
    NvQueue                    *threadQueue;
    NvMediaBool                 exitedFlag;
} ImgCaptureThread;

typedef struct
{
    // Device
    NvMediaDevice              *device;

    // ISC Config
#if BOARD_CODE_NAME == JETSON_PRO_BOARD
    ConfigISCDevices            iscDevices;
    ConfigISCInfo               iscConfigInfo;
    char                        *captureModuleName;
#else
    ExtImgDevice               *extImgDevice;
    ExtImgDevMapInfo            camMap;
#endif

    // ICP Context
    ImageCapture               *icpCtx;

    // Converter
    ImageSurfUtils             *convert;

    // Threads
    ImgCaptureThread            displayThread;

    // General Variables
    CaptureConfigParams        *captureParams;
    NvMediaBool                 quit;
    NvU32                       imagesNum;
    NvU32                       rawBytesPerPixel;

    // EGL
    NvMediaEGLStreamProducer *  eglProducer;
    EGLStreamKHR                eglStream;
    EGLDisplay                  eglDisplay;

    // CUDA consumer
    CUeglStreamConnection       cudaConnection;
    CUgraphicsResource          cudaResource;

    NvMediaSurfaceType          outputSurfType;
    NvU32                       outputWidth;
    NvU32                       outputHeight;

    NvMediaBool                 useHistogramEqualization;
    NvMediaBool                 tpgMode; //intended for capture path validation, shall skip ISC init

} ImgCapture;


NvMediaStatus
ImgCapture_CheckVersion(void);

NvMediaStatus
ImgCapture_Init(ImgCapture **ctx, CaptureConfigParams & captureConfigCollection, NvU32 imagesNum);

void
ImgCapture_Finish(ImgCapture *ctx);

} // namespace ov10635

} // namespace nvidiaio

#endif // USE_CSI_OV10635

#endif // IMGCAPTURE_H
