/*
# Copyright (c) 2014-2016, NVIDIA CORPORATION. All rights reserved.
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

#ifdef USE_NVMEDIA

#include "Private/LogUtils.hpp"

#include <NVX/Application.hpp>
#include <NVX/ProfilerRange.hpp>

#include "FrameSource/NvMedia/NvMediaVideoFrameSourceImpl.hpp"
#include "FrameSource/EGLAPIAccessors.hpp"

#define MAX_ATTRIB 31
#define READ_SIZE (32*1024)
#define COPYFIELD(a, b, field) (a)->field = (b)->field

using namespace nvidiaio::egl_api;

static void SetParamsH264(NVDPictureData *pd, NvMediaPictureInfo *pictureInfo)
{
    NVH264PictureData *h264Data = &pd->CodecSpecific.h264;
    NvMediaPictureInfoH264 *h264PictureInfo = (NvMediaPictureInfoH264 *)pictureInfo;

    h264PictureInfo->field_order_cnt[0] = h264Data->CurrFieldOrderCnt[0];
    h264PictureInfo->field_order_cnt[1] = h264Data->CurrFieldOrderCnt[1];
    h264PictureInfo->is_reference = pd->ref_pic_flag;
    h264PictureInfo->chroma_format_idc = pd->chroma_format;
    COPYFIELD(h264PictureInfo, h264Data, frame_num);
    COPYFIELD(h264PictureInfo, pd, field_pic_flag);
    COPYFIELD(h264PictureInfo, pd, bottom_field_flag);
    COPYFIELD(h264PictureInfo, h264Data, num_ref_frames);
    h264PictureInfo->mb_adaptive_frame_field_flag = h264Data->MbaffFrameFlag;
    COPYFIELD(h264PictureInfo, h264Data, constrained_intra_pred_flag);
    COPYFIELD(h264PictureInfo, h264Data, weighted_pred_flag);
    COPYFIELD(h264PictureInfo, h264Data, weighted_bipred_idc);
    COPYFIELD(h264PictureInfo, h264Data, frame_mbs_only_flag);
    COPYFIELD(h264PictureInfo, h264Data, transform_8x8_mode_flag);
    COPYFIELD(h264PictureInfo, h264Data, chroma_qp_index_offset);
    COPYFIELD(h264PictureInfo, h264Data, second_chroma_qp_index_offset);
    COPYFIELD(h264PictureInfo, h264Data, pic_init_qp_minus26);
    COPYFIELD(h264PictureInfo, h264Data, num_ref_idx_l0_active_minus1);
    COPYFIELD(h264PictureInfo, h264Data, num_ref_idx_l1_active_minus1);
    COPYFIELD(h264PictureInfo, h264Data, log2_max_frame_num_minus4);
    COPYFIELD(h264PictureInfo, h264Data, pic_order_cnt_type);
    COPYFIELD(h264PictureInfo, h264Data, log2_max_pic_order_cnt_lsb_minus4);
    COPYFIELD(h264PictureInfo, h264Data, delta_pic_order_always_zero_flag);
    COPYFIELD(h264PictureInfo, h264Data, direct_8x8_inference_flag);
    COPYFIELD(h264PictureInfo, h264Data, entropy_coding_mode_flag);
    COPYFIELD(h264PictureInfo, h264Data, pic_order_present_flag);
    COPYFIELD(h264PictureInfo, h264Data, deblocking_filter_control_present_flag);
    COPYFIELD(h264PictureInfo, h264Data, redundant_pic_cnt_present_flag);
    COPYFIELD(h264PictureInfo, h264Data, num_slice_groups_minus1);
    COPYFIELD(h264PictureInfo, h264Data, slice_group_map_type);
    COPYFIELD(h264PictureInfo, h264Data, slice_group_change_rate_minus1);
    h264PictureInfo->slice_group_map = h264Data->pMb2SliceGroupMap;
    COPYFIELD(h264PictureInfo, h264Data, fmo_aso_enable);
    COPYFIELD(h264PictureInfo, h264Data, scaling_matrix_present);

    memcpy(h264PictureInfo->scaling_lists_4x4, h264Data->WeightScale4x4, sizeof(h264Data->WeightScale4x4));
    memcpy(h264PictureInfo->scaling_lists_8x8, h264Data->WeightScale8x8, sizeof(h264Data->WeightScale8x8));

    // nvdec specific, not required for avp+vde
    COPYFIELD(h264PictureInfo, pd, nNumSlices);
    COPYFIELD(h264PictureInfo, pd, pSliceDataOffsets);

    for (NvU32 i = 0; i < 16; i++)
    {
        NVH264DPBEntry *dpb_in = &h264Data->dpb[i];
        NvMediaReferenceFrameH264 *dpb_out = &h264PictureInfo->referenceFrames[i];
        nvidiaio::RefCountedFrameBuffer* picbuf = (nvidiaio::RefCountedFrameBuffer*)dpb_in->pPicBuf;

        COPYFIELD(dpb_out, dpb_in, FrameIdx);
        COPYFIELD(dpb_out, dpb_in, is_long_term);
        dpb_out->field_order_cnt[0] = dpb_in->FieldOrderCnt[0];
        dpb_out->field_order_cnt[1] = dpb_in->FieldOrderCnt[1];
        dpb_out->top_is_reference = !!(dpb_in->used_for_reference & 1);
        dpb_out->bottom_is_reference = !!(dpb_in->used_for_reference & 2);
        dpb_out->surface = picbuf ? picbuf->videoSurface : nullptr;
    }
}

// Client callbacks

static void cbRelease(void*, NVDPicBuff *p)
{
    nvidiaio::RefCountedFrameBuffer * buffer = (nvidiaio::RefCountedFrameBuffer*)p;

    if (buffer->nRefs > 0)
        buffer->nRefs--;
}

static NvBool cbDecodePicture(void *ptr, NVDPictureData *pd)
{
    nvidiaio::SampleAppContext *ctx = (nvidiaio::SampleAppContext*)ptr;
    NvMediaStatus status;
    NvMediaPictureInfoH264 picInfoH264;

    if (pd->pCurrPic)
    {
        NvMediaBitstreamBuffer bitStreamBuffer[1];
        SetParamsH264(pd, &picInfoH264);

        nvidiaio::RefCountedFrameBuffer *targetBuffer = (nvidiaio::RefCountedFrameBuffer *)pd->pCurrPic;

        bitStreamBuffer[0].bitstream = (NvU8 *)pd->pBitstreamData;
        bitStreamBuffer[0].bitstreamBytes = pd->nBitstreamDataLen;

        NVXIO_PRINT("DecodePicture %d Ptr:%p Surface:%p (stream ptr:%p size: %d)...",
                ctx->nPicNum, targetBuffer, targetBuffer->videoSurface, pd->pBitstreamData, pd->nBitstreamDataLen);
        ctx->nPicNum++;

        if (targetBuffer->videoSurface)
        {
            status = NvMediaVideoDecoderRender(ctx->decoder, targetBuffer->videoSurface,
                        (NvMediaPictureInfo *)&picInfoH264, 1, &bitStreamBuffer[0]);
            if (status != NVMEDIA_STATUS_OK)
            {
                NVXIO_PRINT("Decode Picture: Decode failed: %d", status);
                return NV_FALSE;
            }
        }
        else
        {
            NVXIO_PRINT("Decode Picture: Invalid target surface");
        }
    }
    else
    {
        NVXIO_PRINT("Decode Picture: No valid frame");
        return NV_FALSE;
    }

    return NV_TRUE;
}

static NvBool cbDisplayPicture(void *ptr, NVDPicBuff *p, NvS64)
{
    nvidiaio::SampleAppContext *ctx = (nvidiaio::SampleAppContext*)ptr;
    nvidiaio::RefCountedFrameBuffer* buffer = (nvidiaio::RefCountedFrameBuffer*)p;

    if (p)
    {
        ctx->frameSource->DisplayFrame(buffer);
    }
    else
    {
        NVXIO_PRINT("Display: Invalid buffer");
        return NV_FALSE;
    }

    return NV_TRUE;
}

static void cbAddRef(void*, NVDPicBuff *p)
{
    nvidiaio::RefCountedFrameBuffer* buffer = (nvidiaio::RefCountedFrameBuffer*)p;
    buffer->nRefs++;
}

static void cbUnhandledNALU(void*, const NvU8*, NvS32)
{
}

static NvS32 cbBeginSequence(void *ptr, const NVDSequenceInfo *pnvsi)
{
    nvidiaio::SampleAppContext *ctx = (nvidiaio::SampleAppContext*)ptr;

    const char* chroma[] =
    {
        "Monochrome",
        "4:2:0",
        "4:2:2",
        "4:4:4"
    };

    auto notify = [&]
    {
        ctx->alive = false;
        // Syncronization
        {
            std::lock_guard<std::mutex> lock(ctx->mutex);
            ctx->isStarted = true;
        }
        ctx->condVariable.notify_one();
    };

    NvU32 decodeBuffers = pnvsi->nDecodeBuffers;
    NvMediaVideoDecoderAttributes attributes;

    if (pnvsi->eCodec != NVCS_H264)
    {
        NVXIO_PRINT("BeginSequence: Invalid codec type: %d", pnvsi->eCodec);
        notify();
        return 0;
    }

    NVXIO_PRINT("BeginSequence: %dx%d (disp: %dx%d) codec: H264 decode buffers: %d aspect: %d:%d fps: %f chroma: %s",
                pnvsi->nCodedWidth, pnvsi->nCodedHeight, pnvsi->nDisplayWidth, pnvsi->nDisplayHeight,
                pnvsi->nDecodeBuffers, pnvsi->lDARWidth, pnvsi->lDARHeight,
                pnvsi->fFrameRate, pnvsi->nChromaFormat > 3 ? "Invalid" : chroma[pnvsi->nChromaFormat]);

    ctx->frameSource->configuration.frameWidth = pnvsi->nDisplayWidth;
    ctx->frameSource->configuration.frameHeight= pnvsi->nDisplayHeight;
    ctx->frameSource->configuration.fps = static_cast<uint>(pnvsi->fFrameRate);
    ctx->frameSource->configuration.format = NVXCU_DF_IMAGE_RGBX;

    if (!ctx->aspectRatio && pnvsi->lDARWidth && pnvsi->lDARHeight)
    {
        double aspect = (float)pnvsi->lDARWidth / (float)pnvsi->lDARHeight;
        if (aspect > 0.3 && aspect < 3.0)
            ctx->aspectRatio = aspect;
    }

    // Check resolution change
    if (pnvsi->nCodedWidth != ctx->decodeWidth || pnvsi->nCodedHeight != ctx->decodeHeight)
    {
        NvMediaVideoCodec codec;
        NvMediaSurfaceType surfType;
        NvU32 maxReferences;

        NVXIO_PRINT("BeginSequence: Resolution changed: Old:%dx%d New:%dx%d",
                    ctx->decodeWidth, ctx->decodeHeight, pnvsi->nCodedWidth, pnvsi->nCodedHeight);

        ctx->decodeWidth = pnvsi->nCodedWidth;
        ctx->decodeHeight = pnvsi->nCodedHeight;

        ctx->displayWidth = pnvsi->nDisplayWidth;
        ctx->displayHeight = pnvsi->nDisplayHeight;

        if (ctx->decoder)
        {
            NvMediaVideoDecoderDestroy(ctx->decoder);
            ctx->decoder = nullptr;
        }

        codec = NVMEDIA_VIDEO_CODEC_H264;

        maxReferences = (decodeBuffers > 0) ? decodeBuffers - 1 : 0;
        maxReferences = (maxReferences > 16) ? 16 : maxReferences;

        NVXIO_PRINT("Create decoder: NVMEDIA_VIDEO_CODEC_H264 Size: %dx%d maxReferences: %d",
                    ctx->decodeWidth, ctx->decodeHeight, maxReferences);
        ctx->decoder = NvMediaVideoDecoderCreate(
            codec,                   // codec
            ctx->decodeWidth,        // width
            ctx->decodeHeight,       // height
            maxReferences,           // maxReferences
            pnvsi->MaxBitstreamSize, //maxBitstreamSize
            5);                      // inputBuffering
        if (!ctx->decoder)
        {
            NVXIO_PRINT("Unable to create decoder");
            notify();
            return 0;
        }

        //set progressive sequence
        attributes.progressiveSequence = pnvsi->bProgSeq;
        NvMediaVideoDecoderSetAttributes(
            ctx->decoder,
            NVMEDIA_VIDEO_DECODER_ATTRIBUTE_PROGRESSIVE_SEQUENCE,
            &attributes);

        for(int i = 0; i < MAX_FRAMES; i++)
        {
            if (ctx->RefFrame[i].videoSurface)
            {
                NvMediaVideoSurfaceDestroy(ctx->RefFrame[i].videoSurface);
            }
        }

        memset(&ctx->RefFrame[0], 0, sizeof(nvidiaio::RefCountedFrameBuffer) * MAX_FRAMES);

        switch (pnvsi->nChromaFormat)
        {
            case 0: // Monochrome
            case 1: // 4:2:0
                NVXIO_PRINT("Chroma format: NvMediaSurfaceType_YV12");
                surfType = NvMediaSurfaceType_YV12;
                break;
            case 2: // 4:2:2
                NVXIO_PRINT("Chroma format: NvMediaSurfaceType_YV16");
                surfType = NvMediaSurfaceType_YV16;
                break;
            case 3: // 4:4:4
                NVXIO_PRINT("Chroma format: NvMediaSurfaceType_YV24");
                surfType = NvMediaSurfaceType_YV24;
                break;
            default:
                NVXIO_PRINT("Invalid chroma format: %d", pnvsi->nChromaFormat);
                notify();
                return 0;
        }

        ctx->producer = NvMediaEglStreamProducerCreate(ctx->device, ctx->eglDisplay, ctx->eglStream,
                                                       ctx->surfaceType, ctx->displayWidth, ctx->displayHeight);
        if(!ctx->producer)
        {
            NVXIO_PRINT("Unable to create producer");
            notify();
            return 0;
        }

        for (int i = 0; i < MAX_RENDER_SURFACE; i++)
        {
            ctx->renderSurfaces[i] = NvMediaVideoSurfaceCreate(
                ctx->device,
                ctx->surfaceType,
                ctx->displayWidth,
                ctx->displayHeight);
            if(!ctx->renderSurfaces[i])
            {
                NVXIO_PRINT("Unable to create render surface");
                notify();
                return 0;
            }
            ctx->freeRenderSurfaces[i] = ctx->renderSurfaces[i];
        }

        ctx->nBuffers = decodeBuffers + MAX_DISPLAY_BUFFERS;

        for(int i = 0; i < ctx->nBuffers; i++)
        {
            ctx->RefFrame[i].videoSurface =
            NvMediaVideoSurfaceCreate(
                ctx->device,
                surfType,
                (pnvsi->nCodedWidth + 15) & ~15,
                                      (pnvsi->nCodedHeight + 15) & ~15);
            if (!ctx->RefFrame[i].videoSurface)
            {
                NVXIO_PRINT("Unable to create video surface");
                notify();
                return 0;
            }
            NVXIO_PRINT("Create video surface[%d]: %dx%d", i,
                        (pnvsi->nCodedWidth + 15) & ~15, (pnvsi->nCodedHeight + 15) & ~15);
            NVXIO_PRINT(" Ptr:%p Surface:%p Device:%p",
                        &ctx->RefFrame[i], ctx->RefFrame[i].videoSurface, ctx->device);
        }

        ctx->frameSource->VideoMixerDestroy();
        ctx->frameSource->VideoMixerInit(ctx->displayWidth, ctx->displayHeight,
                            pnvsi->nCodedWidth, pnvsi->nCodedHeight);
    }
    else
    {
        NVXIO_PRINT("cbBeginSequence: No resolution change");
    }

    // Syncronization
    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        ctx->isStarted = true;
    }
    ctx->condVariable.notify_one();

    return decodeBuffers;
}

static NvBool cbAllocPictureBuffer(void *ptr, NVDPicBuff **p)
{
    NVXIO_PRINT("\tcbAllocPictureBuffer");
    nvidiaio::SampleAppContext *ctx = (nvidiaio::SampleAppContext*)ptr;
    *p = nullptr;

    for (int i = 0; i < ctx->nBuffers; i++)
    {
        if (!ctx->RefFrame[i].nRefs)
        {
            *p = (NVDPicBuff *) &ctx->RefFrame[i];
            ctx->RefFrame[i].nRefs++;
            NVXIO_PRINT("Alloc picture index: %d Ptr:%p Surface:%p", i, *p, ctx->RefFrame[i].videoSurface);
            return NV_TRUE;
        }
    }

    NVXIO_PRINT("Alloc picture failed");
    return NV_FALSE;
}

static NVDClientCb TestClientCb =
{
    &cbBeginSequence,
    &cbDecodePicture,
    &cbDisplayPicture,
    &cbUnhandledNALU,
    &cbAllocPictureBuffer,
    &cbRelease,
    &cbAddRef
};

namespace nvidiaio
{

NvMediaVideoFrameSourceImpl::NvMediaVideoFrameSourceImpl(const std::string & path) :
    FrameSource(nvxio::FrameSource::VIDEO_SOURCE, "NvMediaVideoFrameSource"),
    deviceID(-1),
    exec_target { },
    devMem(nullptr),
    devMemPitch(0)
{
    memset(&context, 0, sizeof(context));

    filePath = path;

    context.eglStream = EGL_NO_STREAM_KHR;
    context.cudaConsumer = 0;
    context.eglDisplay = EGL_NO_DISPLAY;
    context.alive = false;

    context.frameSource = this;

    CUDA_SAFE_CALL( cudaGetDevice(&deviceID) );
    exec_target.base.exec_target_type = NVXCU_STREAM_EXEC_TARGET;
    exec_target.stream = nullptr;
    CUDA_SAFE_CALL( cudaGetDeviceProperties(&exec_target.dev_prop, deviceID) );
}

bool NvMediaVideoFrameSourceImpl::InitializeEGLDisplay()
{
    // Obtain the EGL display
    context.eglDisplay = nvidiaio::EGLDisplayAccessor::getInstance();
    if (context.eglDisplay == EGL_NO_DISPLAY)
    {
        NVXIO_PRINT("EGL failed to obtain display.");
        return false;
    }

    return true;
}

EGLStreamKHR NvMediaVideoFrameSourceImpl::InitializeEGLStream()
{
    static const EGLint streamAttrFIFOMode[] = { EGL_STREAM_FIFO_LENGTH_KHR, 4, EGL_NONE };
    EGLint fifo_length = 4, latency = 0, timeout = 0;
    EGLStreamKHR stream = EGL_NO_STREAM_KHR;

    if (!setupEGLExtensions())
    {
        NVXIO_PRINT("Couldn't setup EGL extensions.");
        return EGL_NO_STREAM_KHR;
    }

    stream = eglCreateStreamKHR(context.eglDisplay, streamAttrFIFOMode);
    if (stream == EGL_NO_STREAM_KHR)
    {
        NVXIO_PRINT("Couldn't create stream.");
        return EGL_NO_STREAM_KHR;
    }

    // Set stream attribute
    if (!eglStreamAttribKHR(context.eglDisplay, stream, EGL_CONSUMER_LATENCY_USEC_KHR, 16000))
    {
        NVXIO_PRINT("Consumer: streamAttribKHR EGL_CONSUMER_LATENCY_USEC_KHR failed");
    }
    if (!eglStreamAttribKHR(context.eglDisplay, stream, EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR, 16000))
    {
        NVXIO_PRINT("Consumer: streamAttribKHR EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR failed");
    }

    // Get stream attributes
    if (!eglQueryStreamKHR(context.eglDisplay, stream, EGL_STREAM_FIFO_LENGTH_KHR, &fifo_length))
    {
        NVXIO_PRINT("Consumer: eglQueryStreamKHR EGL_STREAM_FIFO_LENGTH_KHR failed");
    }
    if (!eglQueryStreamKHR(context.eglDisplay, stream, EGL_CONSUMER_LATENCY_USEC_KHR, &latency))
    {
        NVXIO_PRINT("Consumer: eglQueryStreamKHR EGL_CONSUMER_LATENCY_USEC_KHR failed");
    }
    if (!eglQueryStreamKHR(context.eglDisplay, stream, EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR, &timeout))
    {
        NVXIO_PRINT("Consumer: eglQueryStreamKHR EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR failed");
    }

    NVXIO_PRINT("EGL Stream consumer - Mode: FIFO Length: %d", fifo_length);

    NVXIO_PRINT("EGL stream handle %p", stream);
    NVXIO_PRINT("EGL Stream consumer - Latency: %d usec", latency);
    NVXIO_PRINT("EGL Stream consumer - Timeout: %d usec", timeout);

    return stream;
}

void NvMediaVideoFrameSourceImpl::FinalizeEglStream()
{
    if (context.eglStream != EGL_NO_STREAM_KHR)
    {
        eglDestroyStreamKHR(context.eglDisplay, context.eglStream);
        context.eglStream = EGL_NO_STREAM_KHR;
    }
}


bool NvMediaVideoFrameSourceImpl::InitializeEglCudaConsumer()
{
    if (cudaSuccess != cudaFree(nullptr))
    {
        NVXIO_PRINT("Failed to initialize CUDA context");
        return false;
    }

    NVXIO_PRINT("Connect CUDA consumer");
    CUresult cuStatus = cuEGLStreamConsumerConnect(&context.cudaConsumer, context.eglStream);
    if (CUDA_SUCCESS != cuStatus)
    {
        NVXIO_PRINT("Connect CUDA consumer ERROR %d", cuStatus);
        return false;
    }

    return true;
}

void NvMediaVideoFrameSourceImpl::FinalizeEglCudaConsumer()
{
    if (context.cudaConsumer)
    {
        if (cudaSuccess != cudaFree(nullptr))
        {
            NVXIO_PRINT("Failed to initialize CUDA context");
            return;
        }

        cuEGLStreamConsumerDisconnect(&context.cudaConsumer);
        context.cudaConsumer = 0;
    }
}


bool NvMediaVideoFrameSourceImpl::InitializeDecoder()
{
    float defaultFrameRate = 30.0;
    context.surfaceType =  NvMediaSurfaceType_R8G8B8A8;

    memset(&context.nvsi, 0, sizeof(context.nvsi));

    // create video context
    memset(&context.nvdp, 0, sizeof(NVDParserParams));
    context.nvdp.pClient = &TestClientCb;
    context.nvdp.pClientCtx = &context;
    context.nvdp.lErrorThreshold = 50;
    context.nvdp.lReferenceClockRate = 0;
    context.nvdp.eCodec = NVCS_H264;

    context.ctx = video_parser_create(&context.nvdp);
    if (!context.ctx)
    {
        NVXIO_PRINT("video_parser_create failed");
        return false;
    }

    if (!video_parser_set_attribute(context.ctx,
                               NVDVideoParserAttribute_SetDefaultFramerate,
                               sizeof(float), &defaultFrameRate))
    {
        NVXIO_PRINT("Failed to setup NVDVideoParserAttribute_SetDefaultFramerate");
        return false;
    }

    context.device = NvMediaDeviceCreate();
    if (!context.device)
    {
        NVXIO_PRINT("Unable to create device");
        return false;
    }

    return true;
}

void NvMediaVideoFrameSourceImpl::FinalizeDecoder()
{
    if (context.ctx)
    {
        video_parser_destroy(context.ctx);
        context.ctx = nullptr;
    }
    DisplayFlush();

    for(NvU32 i = 0; i < MAX_FRAMES; i++)
    {
        if (context.RefFrame[i].videoSurface)
        {
            NvMediaVideoSurfaceDestroy(context.RefFrame[i].videoSurface);
            context.RefFrame[i].videoSurface = nullptr;
        }
    }

    if (context.decoder)
    {
        NvMediaVideoDecoderDestroy(context.decoder);
        context.decoder = nullptr;
    }

    DisplayFlush();

    VideoMixerDestroy();

    for (NvU32 i = 0; i < MAX_RENDER_SURFACE; i++)
    {
        if(context.renderSurfaces[i])
        {
            NvMediaVideoSurfaceDestroy(context.renderSurfaces[i]);
            context.renderSurfaces[i] = nullptr;
        }
    }

    if (context.producer)
    {
        NvMediaEglStreamProducerDestroy(context.producer);
        context.producer = nullptr;
    }

    if (context.device)
    {
        NvMediaDeviceDestroy(context.device);
        context.device = nullptr;
    }
}

void NvMediaVideoFrameSourceImpl::FetchVideoFile()
{
    fetchThread = std::thread( [&] ()
    {
        NvU8 * bits = nullptr;
        NvU32 readSize = READ_SIZE;
        FILE * fp = nullptr;

        fp = fopen(filePath.c_str(), "rb");
        if (!fp)
        {
            NVXIO_PRINT("Failed to open stream %s", filePath.c_str());
            return;
        }

        bits = (NvU8*)malloc(readSize);
        if (!bits)
        {
            fclose(fp);

            NVXIO_PRINT("Cannot allocate memory for file reading");
            return;
        }

        context.alive = true;

        while (!feof(fp) && context.alive)
        {
            size_t len;
            bitstream_packet_s packet;

            memset(&packet, 0, sizeof(bitstream_packet_s));

            len = fread(bits, 1, readSize, fp);
            if (len == 0)
            {
                NVXIO_PRINT("Failed to read from the source %s", filePath.c_str());
                context.alive = false;
                fclose(fp);
                free(bits);

                return;
            }

            packet.nDataLength = (NvS32) len;
            packet.pByteStream = bits;
            packet.bEOS = feof(fp);
            packet.bPTSValid = 0;
            packet.llPTS = 0;
            NVXIO_PRINT("flushing");

            if (!video_parser_parse(context.ctx, &packet))
            {
                NVXIO_PRINT("Cannot parse video");
                context.alive = false;
                fclose(fp);
                free(bits);

                return;
            }
        }

        video_parser_flush(context.ctx);
        DisplayFlush();

        free(bits);
        fclose(fp);

        context.alive = false;
    });

    std::unique_lock<std::mutex> lock(context.mutex);
    context.condVariable.wait(lock, [&] () { return context.isStarted; } );
}


bool NvMediaVideoFrameSourceImpl::VideoMixerInit(int width, int height, int videoWidth, int videoHeight)
{
    unsigned int features =  0;
    float aspectRatio = (float)width / (float)height;

    if (context.aspectRatio != 0.0)
    {
        aspectRatio = context.aspectRatio;
    }

    NVXIO_PRINT("VideoMixerInit: %dx%d Aspect: %f", width, height, aspectRatio);

    /* default Deinterlace: Off/Weave */
    NVXIO_PRINT("VideoMixerInit: Surface Renderer Mixer create");
    features |= (NVMEDIA_VIDEO_MIXER_FEATURE_DVD_MIXING_MODE | NVMEDIA_VIDEO_MIXER_FEATURE_SURFACE_RENDERING);

    context.mixer = NvMediaVideoMixerCreate(
        context.device,       // device,
        width,                // mixerWidth
        height,               // mixerHeight
        aspectRatio,          // sourceAspectRatio
        videoWidth,           // primaryVideoWidth
        videoHeight,          // primaryVideoHeight
        0,                    // secondaryVideoWidth
        0,                    // secondaryVideoHeight
        0,                    // graphics0Width
        0,                    // graphics0Height
        0,                    // graphics1Width
        0,                    // graphics1Height
        features,
        nullptr);

    if (!context.mixer)
    {
        NVXIO_PRINT("Unable to create mixer");
        return false;
    }

    NVXIO_PRINT("VideoMixerInit: Mixer:%p", context.mixer);

    return true;
}

void NvMediaVideoFrameSourceImpl::VideoMixerDestroy()
{
    if (context.mixer)
    {
        NvMediaVideoMixerDestroy(context.mixer);
        context.mixer = nullptr;
    }
}

void NvMediaVideoFrameSourceImpl::ReleaseFrame(NvMediaVideoSurface *videoSurface)
{
    for (int i = 0; i < MAX_FRAMES; i++)
    {
        if (videoSurface == context.RefFrame[i].videoSurface)
        {
            cbRelease((void *)&context, (NVDPicBuff *)&context.RefFrame[i]);
            break;
        }
    }
}

void NvMediaVideoFrameSourceImpl::DisplayFlush()
{
    NvMediaVideoSurface *renderSurface = nullptr;

    if (context.producer)
    {
        NVXIO_PRINT("before NvMediaEglStreamProducerGetSurface loop");
        while (NvMediaEglStreamProducerGetSurface(context.producer, &renderSurface, 0) == NVMEDIA_STATUS_OK)
        {
            NVXIO_PRINT("NvMediaEglStreamProducerGetSurface iteration");
            if ((context.surfaceType == NvMediaSurfaceType_R8G8B8A8_BottomOrigin) ||
                (context.surfaceType == NvMediaSurfaceType_R8G8B8A8))
            {
                for (int i = 0; i < MAX_RENDER_SURFACE; i++)
                {
                    if(!context.freeRenderSurfaces[i])
                    {
                        context.freeRenderSurfaces[i] = renderSurface;
                        break;
                    }
                }
            }
            else
            {
                ReleaseFrame(renderSurface);
            }
        }
    }
}


NvMediaVideoSurface * NvMediaVideoFrameSourceImpl::GetRenderSurface()
{
    NvMediaVideoSurface *renderSurface = nullptr;

    if ((context.surfaceType == NvMediaSurfaceType_R8G8B8A8_BottomOrigin) ||
        (context.surfaceType == NvMediaSurfaceType_R8G8B8A8))
    {
        for (int i = 0; i < MAX_RENDER_SURFACE; i++)
        {
            if (context.freeRenderSurfaces[i])
            {
                renderSurface = context.freeRenderSurfaces[i];
                context.freeRenderSurfaces[i] = nullptr;
                return renderSurface;
            }
        }
    }

    while (context.alive)
    {
        NvMediaStatus status = NvMediaEglStreamProducerGetSurface(context.producer, &renderSurface, 50);
        if (status == NVMEDIA_STATUS_ERROR &&
           ((context.surfaceType == NvMediaSurfaceType_R8G8B8A8_BottomOrigin) ||
           (context.surfaceType == NvMediaSurfaceType_R8G8B8A8)))
        {
            NVXIO_PRINT("GetRenderSurface: NvMediaGetSurface waiting");
        }

        // EGL stream producer was able to get a free surface
        if (renderSurface)
        {
            return renderSurface;
        }
    }

    return renderSurface;
}

// push to EGL stream
void NvMediaVideoFrameSourceImpl::DisplayFrame(RefCountedFrameBuffer *frame)
{
    NvMediaPrimaryVideo primaryVideo;
    NvMediaStatus status;
    NvMediaRect primarySourceRect = { 0, 0, (unsigned short)context.displayWidth,
                                            (unsigned short)context.displayHeight };

    if (!frame || !frame->videoSurface)
    {
        NVXIO_PRINT("DisplayFrame: Invalid surface");
        return;
    }

    NvMediaVideoSurface *renderSurface = GetRenderSurface();
    if (!renderSurface)
    {
        if ((context.surfaceType == NvMediaSurfaceType_R8G8B8A8_BottomOrigin) ||
            (context.surfaceType == NvMediaSurfaceType_R8G8B8A8))
        {
            NVXIO_PRINT("DisplayFrame: GetRenderSurface empty");
            return;
        }
    }

    /* Deinterlace Off/Weave */
    primaryVideo.pictureStructure = NVMEDIA_PICTURE_STRUCTURE_FRAME;
    primaryVideo.next = nullptr;
    primaryVideo.current = frame->videoSurface;
    primaryVideo.previous = nullptr;
    primaryVideo.previous2 = nullptr;
    primaryVideo.srcRect = &primarySourceRect;
    primaryVideo.dstRect = nullptr;

    frame->nRefs++;

    if ((context.surfaceType == NvMediaSurfaceType_R8G8B8A8_BottomOrigin) ||
        (context.surfaceType == NvMediaSurfaceType_R8G8B8A8))
    {
        // Render to surface
        NVXIO_PRINT("DisplayFrame: Render to surface");
        status = NvMediaVideoMixerRenderSurface(
            context.mixer, // mixer
            renderSurface, // renderSurface
            nullptr,       // background
            &primaryVideo, // primaryVideo
            nullptr,       // secondaryVideo
            nullptr,       // graphics0
            nullptr);      // graphics1
        if (status != NVMEDIA_STATUS_OK)
        {
            NVXIO_PRINT("DisplayFrame: NvMediaVideoMixerRender failed");
        }
    }

    status = NvMediaEglStreamProducerPostSurface(context.producer,
                                        (context.surfaceType == NvMediaSurfaceType_R8G8B8A8_BottomOrigin) ||
                                        (context.surfaceType == NvMediaSurfaceType_R8G8B8A8) ?
                                        renderSurface : frame->videoSurface,
                                        nullptr);
    if (status != NVMEDIA_STATUS_OK)
    {
        NVXIO_PRINT("DisplayFrame: NvMediaEglStreamProducerPostSurface failed");
    }

    ReleaseFrame((context.surfaceType == NvMediaSurfaceType_R8G8B8A8_BottomOrigin) ||
                         (context.surfaceType == NvMediaSurfaceType_R8G8B8A8) ?
                         primaryVideo.current : renderSurface);
}


bool NvMediaVideoFrameSourceImpl::open()
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "FrameSource::open (NVXIO)");

    close();

    NVXIO_PRINT("Initializing EGL display");
    if (!InitializeEGLDisplay())
    {
        NVXIO_PRINT("Cannot initialize EGL display");
        return false;
    }

    NVXIO_PRINT("Initializing EGL stream");
    context.eglStream = InitializeEGLStream();
    if (context.eglStream == EGL_NO_STREAM_KHR)
    {
        NVXIO_PRINT("Cannot initialize EGL Stream");
        return false;
    }

    NVXIO_PRINT("Initializing EGL consumer");
    if (!InitializeEglCudaConsumer())
    {
        NVXIO_PRINT("Cannot initialize CUDA consumer");
        return false;
    }

    NVXIO_PRINT("Initializing NvMedia Decoder");
    if (!InitializeDecoder())
    {
        NVXIO_PRINT("Cannot initialize NvMedia decoder");
        return false;
    }

    // fetching
    FetchVideoFile();

    return true;
}

void convertFrame(nvxcu_stream_exec_target_t &exec_target,
                  const image_t & image,
                  const FrameSource::Parameters & configuration,
                  int width, int height,
                  bool usePitch, size_t pitch,
                  int depth, void * decodedPtr,
                  bool is_cuda,
                  void *& devMem,
                  size_t & devMemPitch);

vx_image wrapNVXIOImage(vx_context context,
                        const image_t & image);

FrameSource::FrameStatus NvMediaVideoFrameSourceImpl::fetch(const image_t & image, uint32_t timeout /*milliseconds*/)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "FrameSource::fetch (NVXIO)");

    if (cudaSuccess != cudaFree(nullptr))
    {
        NVXIO_PRINT("Failed to initialize CUDA context");
        return nvxio::FrameSource::CLOSED;
    }

    CUresult cuStatus;
    CUgraphicsResource cudaResource;

    EGLint streamState = 0;
    if (!eglQueryStreamKHR(context.eglDisplay, context.eglStream, EGL_STREAM_STATE_KHR, &streamState))
    {
        NVXIO_PRINT("Cuda consumer, eglQueryStreamKHR EGL_STREAM_STATE_KHR failed");
        return nvxio::FrameSource::FrameStatus::CLOSED;
    }

    if (streamState == EGL_STREAM_STATE_DISCONNECTED_KHR)
    {
        NVXIO_PRINT("CUDA Consumer: - EGL_STREAM_STATE_DISCONNECTED_KHR received");
        return nvxio::FrameSource::FrameStatus::CLOSED;
    }

    if (streamState != EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR)
    {
        return nvxio::FrameSource::TIMEOUT;
    }

    cuStatus = cuEGLStreamConsumerAcquireFrame(&context.cudaConsumer, &cudaResource, nullptr, timeout);
    if (cuStatus != CUDA_SUCCESS)
    {
        NVXIO_PRINT("Cuda Acquire failed cuStatus=%d", cuStatus);

        return nvxio::FrameSource::FrameStatus::TIMEOUT;
    }

    CUeglFrame eglFrame;
    cuStatus = cuGraphicsResourceGetMappedEglFrame(&eglFrame, cudaResource, 0, 0);
    if (cuStatus != CUDA_SUCCESS)
    {
        const char* error;
        cuGetErrorString(cuStatus, &error);
        NVXIO_PRINT("Cuda get resource failed with error: \"%s\"", error);
        return nvxio::FrameSource::FrameStatus::CLOSED;
    }

    NVXIO_ASSERT(eglFrame.width == configuration.frameWidth);
    NVXIO_ASSERT(eglFrame.height == configuration.frameHeight);

    NVXIO_ASSERT(configuration.format == NVXCU_DF_IMAGE_RGBX);
    NVXIO_ASSERT(eglFrame.pitch == ((eglFrame.width * 4 + 3) >> 2) << 2);
    convertFrame(exec_target,
                 image,
                 configuration,
                 eglFrame.width, eglFrame.height,
                 true, eglFrame.pitch,
                 4, eglFrame.frame.pPitch[0],
                 true,
                 devMem,
                 devMemPitch);

    cuStatus = cuEGLStreamConsumerReleaseFrame(&context.cudaConsumer, cudaResource, nullptr);
    if (cuStatus != CUDA_SUCCESS)
    {
        NVXIO_PRINT("Failed to release EGL frame");
        close();
        return nvxio::FrameSource::FrameStatus::CLOSED;
    }

    return nvxio::FrameSource::FrameStatus::OK;
}

FrameSource::Parameters NvMediaVideoFrameSourceImpl::getConfiguration()
{
    return configuration;
}

bool NvMediaVideoFrameSourceImpl::setConfiguration(const FrameSource::Parameters& params)
{
    NVXIO_ASSERT(!context.alive);

    bool result = true;

    // ignore FPS, width, height values
    if (params.frameWidth != (uint32_t)-1)
        result = false;
    if (params.frameHeight != (uint32_t)-1)
        result = false;
    if (params.fps != (uint32_t)-1)
        result = false;

    NVXIO_ASSERT((params.format == NVXCU_DF_IMAGE_NV12) ||
                 (params.format == NVXCU_DF_IMAGE_U8) ||
                 (params.format == NVXCU_DF_IMAGE_RGB) ||
                 (params.format == NVXCU_DF_IMAGE_RGBX)||
                 (params.format == NVXCU_DF_IMAGE_NONE));


    configuration.format = params.format;

    return result;
}

void NvMediaVideoFrameSourceImpl::close()
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "FrameSource::close (NVXIO)");

    context.alive = false;
    context.isStarted = false;

    if (fetchThread.joinable())
        fetchThread.join();

    NVXIO_PRINT("Finalizing EGL CUDA consumer");
    FinalizeEglCudaConsumer();

    NVXIO_PRINT("Finalizing NvMedia decoder");
    FinalizeDecoder();

    NVXIO_PRINT("Finalizing EGL Stream");
    FinalizeEglStream();

    context.decodeWidth = -1;
    context.decodeHeight = -1;

    if (devMem)
    {
        cudaFree(devMem);
        devMem = nullptr;
    }
}

NvMediaVideoFrameSourceImpl::~NvMediaVideoFrameSourceImpl()
{
    close();
}

} // namespace nvidiaio

#endif // USE_NVMEDIA
