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

#include <memory>

#include <OVX/RenderOVX.hpp>
#include <NVX/Application.hpp>

#include "Render/EventLogger.hpp"
#include "Render/EventPlayer.hpp"

#ifdef USE_GUI
# include "Render/GlfwUIRenderImpl.hpp"
# ifdef USE_GSTREAMER
#  include "Render/GStreamer/GStreamerVideoRenderImpl.hpp"
#  include "Render/GStreamer/GStreamerImagesRenderImpl.hpp"
# endif
# ifdef USE_OPENCV
#  include "Render/OpenCV/OpenGLOpenCVRenderImpl.hpp"
# endif
#endif

#include "Render/StubRenderImpl.hpp"
#include "Wrappers/RenderOVXWrapper.hpp"
#include "Render/Wrappers/RenderWrapper.hpp"

namespace ovxio
{

std::unique_ptr<Render> createVideoRender(vx_context context, const std::string& path,
                                          vx_uint32 width, vx_uint32 height, vx_uint32 format)
{
    std::unique_ptr<nvidiaio::Render> ptr =
            nvidiaio::createVideoRender(path, width, height, static_cast<nvxcu_df_image_e>(format));

    if (!ptr)
        return nullptr;

    return ovxio::makeUP<RenderWrapper>(context, std::move(ptr));
}

std::unique_ptr<Render> createImageRender(vx_context context, const std::string& path,
                                          vx_uint32 width, vx_uint32 height, vx_uint32 format)
{
    std::unique_ptr<nvidiaio::Render> ptr =
            nvidiaio::createImageRender(path, width, height, static_cast<nvxcu_df_image_e>(format));

    if (!ptr)
        return nullptr;

    return ovxio::makeUP<RenderWrapper>(context, std::move(ptr));
}

std::unique_ptr<Render> createWindowRender(vx_context context, const std::string& title, vx_uint32 width, vx_uint32 height,
                                           vx_uint32 format, bool doScale, bool fullscreen)
{
    std::unique_ptr<nvidiaio::Render> ptr =
            nvidiaio::createWindowRender(title, width, height, static_cast<nvxcu_df_image_e>(format),
                                         doScale, fullscreen);

    if (!ptr)
        return nullptr;

    return makeUP<RenderWrapper>(context, std::move(ptr));
}

std::unique_ptr<Render> createDefaultRender(vx_context context, const std::string& title, vx_uint32 width, vx_uint32 height,
                                            vx_uint32 format, bool doScale, bool fullscreen)
{
    std::unique_ptr<nvidiaio::Render> ptr =
            nvidiaio::createDefaultRender(title, width, height, static_cast<nvxcu_df_image_e>(format),
                                          doScale, fullscreen);

    if (!ptr)
        return nullptr;

    return makeUP<RenderWrapper>(context, std::move(ptr));
}

} // namespace ovxio
