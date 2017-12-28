/* Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef NVMEDIA_IPP_MAIN_HPP
#define NVMEDIA_IPP_MAIN_HPP

#ifdef USE_CSI_OV10640

#include <stdlib.h>
#include <string.h>

#include "nvcommon.h"
#include "nvmedia.h"
#include "nvmedia_image.h"
#include "nvmedia_isp.h"
#include "thread_utils.h"

#include "FrameSource/NvMedia/NvMediaCameraConfigParams.hpp"

#include <string>

#define MAX_STRING_SIZE                 256
#define MAX_CONFIG_SECTIONS             20

typedef enum
{
    NVMEDIA_NOACPLUGIN,
    NVMEDIA_SIMPLEACPLUGIN,
    NVMEDIA_NVACPLUGIN
} NvMediaACPluginType;

#endif // USE_CSI_OV10640

#endif // NVMEDIA_IPP_MAIN_HPP
