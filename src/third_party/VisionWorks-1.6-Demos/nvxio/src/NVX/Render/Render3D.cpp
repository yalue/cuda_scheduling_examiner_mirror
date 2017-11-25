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

#include <memory>

#include <OVX/Render3DOVX.hpp>
#include <NVX/Application.hpp>

#ifdef USE_GUI
# include "Render/CUDA-OpenGL/BaseRender3DImpl.hpp"
#endif

#include "Render/Wrappers/Render3DWrapper.hpp"

namespace nvidiaio
{

std::unique_ptr<Render3D> createDefaultRender3D(int32_t xPos, int32_t yPos,
                                                const std::string& title, uint32_t width, uint32_t height)
{
    if (nvxio::Application::get().getPreferredRenderName() == "default")
    {
#ifdef USE_GUI
        std::unique_ptr<BaseRender3DImpl> render(new BaseRender3DImpl());
        if (!render->open(xPos, yPos, width, height, title))
        {
            return nullptr;
        }

        return std::unique_ptr<Render3D>(std::move(render));
#else
        (void)xPos;
        (void)yPos;
        (void)title;
        (void)width;
        (void)height;
#endif
    }

    return nullptr;
}

} // namespace nvidiaio


namespace nvxio
{
#include "NVX/Utility.hpp"

std::unique_ptr<Render3D> createDefaultRender3D(int32_t xPos, int32_t yPos, const std::string& title,
                                                uint32_t width, uint32_t height)
{
    std::unique_ptr<nvidiaio::Render3D> ptr = nvidiaio::createDefaultRender3D(xPos, yPos, title, width, height);

    if (!ptr)
        return nullptr;

    return nvxio::makeUP<Render3DWrapper>(std::move(ptr));
}

} // namespace nvxio
