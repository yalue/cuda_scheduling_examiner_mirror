/*
 * Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef NVMEDIA_IPP_COMPONENT_HPP
#define NVMEDIA_IPP_COMPONENT_HPP

#ifdef USE_CSI_OV10640

#include "ipp_raw.hpp"
#include "buffer_utils.h"

NvMediaStatus
IPPSetCaptureSettings (
    IPPCtx *ctx,
    CaptureConfigParams *config);

// Create Raw Pipeline
NvMediaStatus IPPCreateRawPipeline(IPPCtx *ctx);

#endif // USE_CSI_OV10640

#endif // NVMEDIA_IPP_COMPONENT_HPP
