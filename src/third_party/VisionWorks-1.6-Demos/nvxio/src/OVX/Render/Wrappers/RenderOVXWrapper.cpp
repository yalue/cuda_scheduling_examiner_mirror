/*
# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

#include "RenderOVXWrapper.hpp"

#include <OVX/UtilityOVX.hpp>

#include <cstring>

namespace ovxio
{

RenderWrapper::RenderWrapper(vx_context context, std::unique_ptr<nvidiaio::Render> render) :
    Render(static_cast<TargetType>(render->getTargetType()),
           render->getRenderName()),
    render_(std::move(render)), vxContext(context)
{
    NVXIO_ASSERT(render_);
}

#ifndef __ANDROID__

void RenderWrapper::setOnKeyboardEventCallback(OnKeyboardEventCallback callback, void * context)
{
    render_->setOnKeyboardEventCallback(
                reinterpret_cast<nvidiaio::Render::OnKeyboardEventCallback>(callback), context);
}

void RenderWrapper::setOnMouseEventCallback(OnMouseEventCallback callback, void * context)
{
    render_->setOnMouseEventCallback(
                reinterpret_cast<nvidiaio::Render::OnMouseEventCallback>(callback), context);
}

#endif // __ANDROID__

void RenderWrapper::putImage(vx_image image)
{
    ovxio::image_t mapper(image, VX_READ_ONLY, NVX_MEMORY_TYPE_CUDA);
    render_->putImage(mapper);
}

void RenderWrapper::putTextViewport(const std::string & text, const TextBoxStyle & style)
{
    render_->putTextViewport(text, *(nvidiaio::Render::TextBoxStyle *)&style);
}

void RenderWrapper::putFeatures(vx_array location, const FeatureStyle & style)
{
    ovxio::array_t mapper(location, VX_READ_ONLY, NVX_MEMORY_TYPE_CUDA);
    render_->putFeatures(mapper,
                         *(nvidiaio::Render::FeatureStyle *)&style);
}

void RenderWrapper::putFeatures(vx_array location, vx_array styles)
{
    ovxio::array_t locationMapper(location, VX_READ_ONLY, NVX_MEMORY_TYPE_CUDA),
            stylesMapper(styles, VX_READ_ONLY, NVX_MEMORY_TYPE_CUDA);
    render_->putFeatures(locationMapper, stylesMapper);
}

void RenderWrapper::putLines(vx_array lines, const LineStyle & style)
{
    ovxio::array_t mapper(lines, VX_READ_ONLY, NVX_MEMORY_TYPE_CUDA);
    render_->putLines(mapper,
                      *(nvidiaio::Render::LineStyle *)&style);
}

void RenderWrapper::putConvexPolygon(vx_array vertices, const LineStyle & style)
{
    ovxio::array_t mapper(vertices, VX_READ_ONLY, NVX_MEMORY_TYPE_CUDA);
    render_->putConvexPolygon(mapper,
                              *(nvidiaio::Render::LineStyle *)&style);
}

void RenderWrapper::putMotionField(vx_image field, const MotionFieldStyle & style)
{
    ovxio::image_t mapper(field, VX_READ_ONLY, NVX_MEMORY_TYPE_CUDA);
    render_->putMotionField(mapper,
                            *(nvidiaio::Render::MotionFieldStyle *)&style);
}

void RenderWrapper::putObjectLocation(const vx_rectangle_t & location, const DetectedObjectStyle & style)
{
    render_->putObjectLocation(*(const nvxcu_rectangle_t *)&location,
                               *(nvidiaio::Render::DetectedObjectStyle *)&style);
}

void RenderWrapper::putCircles(vx_array circles, const CircleStyle & style)
{
    ovxio::array_t mapper(circles, VX_READ_ONLY, NVX_MEMORY_TYPE_CUDA);
    render_->putCircles(mapper,
                        *(nvidiaio::Render::CircleStyle *)&style);
}

void RenderWrapper::putArrows(vx_array old_points, vx_array new_points, const LineStyle & style)
{
    ovxio::array_t oldPointsMapper(old_points, VX_READ_ONLY, NVX_MEMORY_TYPE_CUDA),
            newPointsMapper(new_points, VX_READ_ONLY, NVX_MEMORY_TYPE_CUDA);
    render_->putArrows(oldPointsMapper, newPointsMapper,
                       *(nvidiaio::Render::LineStyle *)&style);
}

#ifndef __ANDROID__

bool RenderWrapper::flush()
{
    return render_->flush();
}

void RenderWrapper::close()
{
    render_->close();
}

#endif // __ANDROID__

vx_uint32 RenderWrapper::getViewportWidth() const
{
    return render_->getViewportWidth();
}

vx_uint32 RenderWrapper::getViewportHeight() const
{
    return render_->getViewportHeight();
}

RenderWrapper::~RenderWrapper()
{
#ifndef __ANDROID__
    close();
#endif
}

} // namespace ovxio
