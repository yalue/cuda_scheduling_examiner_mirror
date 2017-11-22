/* Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef NVMEDIA_IPP_RAW_HPP
#define NVMEDIA_IPP_RAW_HPP

#ifdef USE_CSI_OV10640

#include <cudaEGL.h>

#include "buffer_utils.h"
#include "log_utils.h"
#include "misc_utils.h"
#include "nvmedia_ipp.h"
#include "nvmedia_eglstream.h"
#include "nvmedia_icp.h"
#include "nvmedia_isc.h"
#include "nvmedia_2d.h"

#include "thread_utils.h"
#include "img_dev.h"
#include "nvltm.h"

#include "main.hpp"
#include "eglstrm_setup.hpp"

#define IMAGE_BUFFERS_POOL_SIZE        3
#define STATS_BUFFERS_POOL_SIZE        3
#define SENSOR_BUFFERS_POOL_SIZE       3
#define MAX_CAPTURE_BUFFERS            2

#define ISP_STATS_NUM_BINS             256
#define ISP_STATS_HEIGHT_PERCENTAGE    12
#define ISP_STATS_RANGE_LOW            0
#define ISP_STATS_RANGE_HIGH           8192

#define ENQUEUE_TIMEOUT                100
#define FEED_FRAME_TIMEOUT             100
#define WAIT_FOR_IDLE_TIMEOUT          100
#define MAX_ERROR_QUEUE_SIZE           20

typedef struct {
    void   *ctx; // pointer to IPPTest
    NvU32   id;
} ThreadData;

typedef struct {
    NvMediaBool captureEnabled;
    NvMediaBool ispEnabled[NVMEDIA_MAX_AGGREGATE_IMAGES];
    NvMediaBool outputEnabled[NVMEDIA_MAX_AGGREGATE_IMAGES];
    NvMediaBool controlAlgorithmEnabled[NVMEDIA_MAX_AGGREGATE_IMAGES];

    char                       *ispConfigFile;
    NvMediaDevice              *device;
    NvMediaIPADevice           *ipaDevice;

    // IPPManager
    NvMediaIPPManager          *ippManager;
    // IPP
    NvMediaIPPPipeline         *ipp[NVMEDIA_MAX_PIPELINES_PER_MANAGER];

    // IPP component
    NvMediaIPPComponent *ippComponents[NVMEDIA_MAX_PIPELINES_PER_MANAGER][NVMEDIA_MAX_COMPONENTS_PER_PIPELINE];
    // IPP ISP component
    NvMediaIPPComponent *ippIspComponents[NVMEDIA_MAX_PIPELINES_PER_MANAGER];
    // IPP ISC component
    NvMediaIPPComponent *ippIscComponents[NVMEDIA_MAX_PIPELINES_PER_MANAGER];
    // IPP Control Algorithm component
    NvMediaIPPComponent *ippControlAlgorithmComponents[NVMEDIA_MAX_PIPELINES_PER_MANAGER];
    // IPP component Num
    NvU32 componentNum[NVMEDIA_MAX_PIPELINES_PER_MANAGER];
    // IPP output component
    NvMediaIPPComponent        *outputComponent[NVMEDIA_MAX_PIPELINES_PER_MANAGER];
    NvMediaBool                 getIPPOutputThreadExited;
    NvThread                   *getOutputThread;
    // IPP component paramenters
    // ICP
    NvMediaICPSettings          captureSettings;

    // ISC
    ExtImgDevice               *extImgDevice;
    ExtImgDevMapInfo           camMap;

    NvMutex                    *processorMutex;

    NvMediaBool                 quit;
    NvMediaIPPRawCompressionFormat rawCompressionFormat;
    NvU32                       inputFormatWidthMultiplier;
    NvMediaSurfaceType          inputSurfType;
    NvU32                       inputSurfAttributes;
    NvMediaImageAdvancedConfig  inputSurfAdvConfig;
    NvU32                       inputWidth;
    NvU32                       inputHeight;

    NvMediaSurfaceType          outputSurfType;

    NvU32                       displayId;

    NvMediaBool                 useOffsetsFlag;
    NvU32                       rawBytesPerPixel;
    NvMediaBool                 aggregateFlag;
    NvU32                       imagesNum;
    NvU32                       ippNum;
    NvMediaSurfaceType          ispOutType;
    NvU32                       framesProcessed;
    NvMediaBool                 showTimeStamp;
    NvMediaSurfaceType          eglSurfaceType;
    NvMediaBool                 showMetadataFlag;
    NvMediaACPluginType         pluginFlag;

    // EGL stuff
    NvMediaEGLStreamProducer *  eglProducer;
    EGLStreamKHR                eglStream[NVMEDIA_MAX_AGGREGATE_IMAGES];
    EGLDisplay                  eglDisplay;

    // CUDA consumer
    CUeglStreamConnection       cudaConnection[NVMEDIA_MAX_AGGREGATE_IMAGES];

} IPPCtx;

typedef NvMediaStatus (* ProcessorFunc) (IPPCtx *ctx);

/*  IPPInit
    IPPInit()  Create IPP pipelines, allocate needed
    components and build pipelines between components
*/
NvMediaStatus
IPPInit (IPPCtx *ctx,
         const CaptureConfigParams & captureParams);

/*  IPPStart
    IPPStart()  Starts IPP pipelines.
*/
NvMediaStatus
IPPStart (IPPCtx *ctx);

/*  IPPStop
    IPPStop()  Stop the IPP pipeline
*/
NvMediaStatus
IPPStop (IPPCtx *ctx);

/*  IPPFini
    IPPFini()  Destory the IPP pipeline, and
    release the resources used
*/
NvMediaStatus
IPPFini (IPPCtx *ctx);

#endif // USE_CSI_OV10640

#endif // NVMEDIA_IPP_RAW_HPP
