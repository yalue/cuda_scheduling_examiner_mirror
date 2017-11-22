/* Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef NVMEDIA_IMAGEPRODUCER_HPP
#define NVMEDIA_IMAGEPRODUCER_HPP

#ifdef USE_CSI_OV10640

#include "eglstrm_setup.hpp"
#include "interop.hpp"

#define EGL_PRODUCER_TIMEOUT_MS    16
#define GET_FRAME_TIMEOUT          500

// The max number of retries after ProducerGetImage()
// times out, which happens because consumer
// might be slower than producer.
#define EGL_PRODUCER_GET_IMAGE_MAX_RETRIES 1000

typedef struct {
    NvMediaDevice              *device;
    char                       *inputImages;
    NvU32                       width;
    NvU32                       height;
    NvU32                       ippNum;
    NvU32                       frameCount;
    NvMediaSurfaceType          surfaceType;
    NvMediaIPPComponent        *outputComponent[NVMEDIA_MAX_AGGREGATE_IMAGES];
    NvMediaBool                 eglProducerGetImageFlag[NVMEDIA_MAX_AGGREGATE_IMAGES];
    //EGL params
    NvMediaEGLStreamProducer   *eglProducer[NVMEDIA_MAX_AGGREGATE_IMAGES];
    EGLStreamKHR                eglStream[NVMEDIA_MAX_AGGREGATE_IMAGES];
    EGLDisplay                  eglDisplay;
    NvMediaBool                *producerExited;
    NvMediaBool                *quit;
    NvMediaBool                 showTimeStamp;
    NvMediaBool                 showMetadataFlag;
    NvMediaBool                 fifoMode;

} ImageProducerCtx;

/* Intialize IPP Image Producer context and create producer*/
ImageProducerCtx*
ImageProducerInit(NvMediaDevice *device,
                  EglStreamClient *streamClient,
                  NvU32 width, NvU32 height,
                  InteropContext *interopCtx);
/* Clears IPP Image Producer context */
NvMediaStatus ImageProducerFini(ImageProducerCtx *client);

/* ImageProducerProc() is the IPPs output thread frunction.
   It is waiting on the output from IPPs, and put the output
   images into the input queue.*/
void ImageProducerProc(void *data, void *user_data);

#endif // USE_CSI_OV10640

#endif // NVMEDIA_IMAGEPRODUCER_HPP
