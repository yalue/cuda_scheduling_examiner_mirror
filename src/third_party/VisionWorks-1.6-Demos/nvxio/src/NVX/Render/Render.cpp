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
#include "Render/Wrappers/RenderWrapper.hpp"

namespace nvidiaio
{

static std::string patchWindowTitle(const std::string & windowTitle)
{
    std::string retVal = windowTitle;

    std::replace(retVal.begin(), retVal.end(), '/', '|');
    std::replace(retVal.begin(), retVal.end(), '\\', '|');

    return retVal;
}

static std::unique_ptr<Render> createSmartRender(std::unique_ptr<Render> specializedRender)
{
    nvxio::Application &app = nvxio::Application::get();
    std::unique_ptr<Render> render = std::move(specializedRender);

    if (!app.getScenarioName().empty())
    {
        std::unique_ptr<EventPlayer> player(new EventPlayer);
        if (player->init(app.getScenarioName(), app.getScenarioLoopCount()))
        {
            // To read data for the first frame
            player->flush();
            player->setEfficientRender(std::move(render));
            render = std::move(player);
        }
        else
        {
            NVXIO_THROW_EXCEPTION("Warning: cannot open scenario \"" << app.getScenarioName() << "\"");
        }
    }

    if (!app.getEventLogName().empty())
    {
        std::unique_ptr<EventLogger> logger(new EventLogger(app.getEventLogDumpFramesFlag()));
        if (logger->init(app.getEventLogName()))
        {
            logger->setEfficientRender(std::move(render));
            render = std::move(logger);
        }
        else
        {
            fprintf(stderr, "Warning: cannot open log file \"%s\"\n", app.getEventLogName().c_str());
        }
    }

    return render;
}

std::unique_ptr<Render> createVideoRender(const std::string& path, uint32_t width, uint32_t height, nvxcu_df_image_e format)
{
#if defined USE_GUI && defined USE_GSTREAMER
    std::unique_ptr<GStreamerVideoRenderImpl> gst_render(new GStreamerVideoRenderImpl());

    if (!gst_render->open(path, width, height, format))
        return nullptr;

    return createSmartRender(std::move(gst_render));
#elif defined USE_GUI && defined USE_OPENCV
    std::unique_ptr<OpenGLOpenCVRenderImpl> ocv_render(new OpenGLOpenCVRenderImpl(
        nvxio::Render::VIDEO_RENDER, "OpenGLOpenCVVideoRenderImpl"));

    if (!ocv_render->open(path, width, height, format))
        return nullptr;

    return createSmartRender(std::move(ocv_render));
#else
    (void)path;
    (void)width;
    (void)height;
    (void)format;
#endif
    return nullptr;
}

std::unique_ptr<Render> createImageRender(const std::string& path, uint32_t width, uint32_t height, nvxcu_df_image_e format)
{
#if defined USE_GUI && defined USE_GSTREAMER
    std::unique_ptr<GStreamerImagesRenderImpl> gst_render(new GStreamerImagesRenderImpl());

    if (!gst_render->open(path, width, height, format))
        return nullptr;

    return createSmartRender(std::move(gst_render));
#elif defined USE_GUI && defined USE_OPENCV
    std::unique_ptr<OpenGLOpenCVRenderImpl> ocv_render(new OpenGLOpenCVRenderImpl(
        nvxio::Render::IMAGE_RENDER, "OpenGLOpenCVImagesRenderImpl"));

    if (!ocv_render->open(path, width, height, format))
        return nullptr;

    return createSmartRender(std::move(ocv_render));
#else
    (void)path;
    (void)width;
    (void)height;
    (void)format;
#endif
    return nullptr;
}

std::unique_ptr<Render> createWindowRender(const std::string& title, uint32_t width, uint32_t height,
                                           nvxcu_df_image_e format, bool doScale, bool fullscreen)
{
#ifdef USE_GUI
    std::unique_ptr<GlfwUIImpl> render(new GlfwUIImpl(nvxio::Render::WINDOW_RENDER, "GlfwOpenGlRender"));

    if (!render->open(title, width, height, format, doScale, fullscreen))
        return nullptr;

    return createSmartRender(std::move(render));
#else
    (void)title;
    (void)width;
    (void)height;
    (void)format;
    (void)doScale;
    (void)fullscreen;

    return nullptr;
#endif
}

std::unique_ptr<Render> createDefaultRender(const std::string& title, uint32_t width, uint32_t height,
                                            nvxcu_df_image_e format, bool doScale, bool fullscreen)
{
    std::string prefferedRenderName = nvxio::Application::get().getPreferredRenderName();

    if (prefferedRenderName == "default")
    {
        std::unique_ptr<Render> render = createWindowRender(title, width, height, format, doScale, fullscreen);

        if (!render)
            render = createVideoRender(title + ".avi", width, height, format);

        return render;
    }
    else if (prefferedRenderName == "window")
    {
        return createWindowRender(title, width, height, format, doScale, fullscreen);
    }
    else if (prefferedRenderName == "video")
    {
        return createVideoRender(patchWindowTitle(title + ".avi"), width, height, format);
    }
    else if (prefferedRenderName == "image")
    {
        return createImageRender(patchWindowTitle(title + "_%05d.png"), width, height, format);
    }
    else if (prefferedRenderName == "stub")
    {
        std::unique_ptr<StubRenderImpl> render(new StubRenderImpl());
        NVXIO_ASSERT(render->open(title, width, height, format));
        return createSmartRender(std::move(render));
    }

    return nullptr;
}

} // namespace nvidiaio


namespace nvxio
{
#include "NVX/Utility.hpp"

std::unique_ptr<Render> createVideoRender(const std::string& path,
                                          uint32_t width, uint32_t height, nvxcu_df_image_e format)
{
    std::unique_ptr<nvidiaio::Render> ptr =
            nvidiaio::createVideoRender(path, width, height, format);

    if (!ptr)
        return nullptr;

    return nvxio::makeUP<RenderWrapper>(std::move(ptr));
}

std::unique_ptr<Render> createImageRender(const std::string& path,
                                          uint32_t width, uint32_t height, nvxcu_df_image_e format)
{
    std::unique_ptr<nvidiaio::Render> ptr =
            nvidiaio::createImageRender(path, width, height, format);

    if (!ptr)
        return nullptr;

    return nvxio::makeUP<RenderWrapper>(std::move(ptr));
}

std::unique_ptr<Render> createWindowRender(const std::string& title, uint32_t width, uint32_t height,
                                           nvxcu_df_image_e format, bool doScale, bool fullscreen)
{
    std::unique_ptr<nvidiaio::Render> ptr =
            nvidiaio::createWindowRender(title, width, height, format,
                                         doScale, fullscreen);

    if (!ptr)
        return nullptr;

    return nvxio::makeUP<RenderWrapper>(std::move(ptr));
}

std::unique_ptr<Render> createDefaultRender(const std::string& title, uint32_t width, uint32_t height,
                                            nvxcu_df_image_e format, bool doScale, bool fullscreen)
{
    std::unique_ptr<nvidiaio::Render> ptr =
            nvidiaio::createDefaultRender(title, width, height, format,
                                          doScale, fullscreen);

    if (!ptr)
        return nullptr;

    return nvxio::makeUP<RenderWrapper>(std::move(ptr));
}

} // namespace nvxio
