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

#include "RenderWrapper.hpp"


#include <cstring>

namespace nvxio
{

RenderWrapper::RenderWrapper(std::unique_ptr<nvidiaio::Render> render) :
    Render(static_cast<TargetType>(render->getTargetType()),
           render->getRenderName()),
    render_(std::move(render))
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

void RenderWrapper::putImage(const nvxcu_pitch_linear_image_t & image)
{
    render_->putImage(nvidiaio::image_t(image));
}

void RenderWrapper::putTextViewport(const std::string & text, const TextBoxStyle & style)
{
    render_->putTextViewport(text, *(nvidiaio::Render::TextBoxStyle *)&style);
}

void RenderWrapper::putFeatures(const nvxcu_plain_array_t & location, const FeatureStyle & style)
{
    render_->putFeatures(nvidiaio::array_t(location),
                         *(nvidiaio::Render::FeatureStyle *)&style);
}

void RenderWrapper::putFeatures(const nvxcu_plain_array_t & location, const nvxcu_plain_array_t & styles)
{
    render_->putFeatures(nvidiaio::array_t(location), nvidiaio::array_t(styles));
}

void RenderWrapper::putLines(const nvxcu_plain_array_t & lines, const LineStyle & style)
{
    render_->putLines(nvidiaio::array_t(lines),
                      *(nvidiaio::Render::LineStyle *)&style);
}

void RenderWrapper::putConvexPolygon(const nvxcu_plain_array_t & vertices, const LineStyle & style)
{
    render_->putConvexPolygon(nvidiaio::array_t(vertices),
                              *(nvidiaio::Render::LineStyle *)&style);
}

void RenderWrapper::putMotionField(const nvxcu_pitch_linear_image_t & field, const MotionFieldStyle & style)
{
    render_->putMotionField(nvidiaio::image_t(field),
                            *(nvidiaio::Render::MotionFieldStyle *)&style);
}

void RenderWrapper::putObjectLocation(const nvxcu_rectangle_t & location, const DetectedObjectStyle & style)
{
    render_->putObjectLocation(*(const nvxcu_rectangle_t *)&location,
                               *(nvidiaio::Render::DetectedObjectStyle *)&style);
}

void RenderWrapper::putCircles(const nvxcu_plain_array_t & circles, const CircleStyle & style)
{
    render_->putCircles(nvidiaio::array_t(circles),
                        *(nvidiaio::Render::CircleStyle *)&style);
}

void RenderWrapper::putArrows(const nvxcu_plain_array_t & old_points, const nvxcu_plain_array_t & new_points, const LineStyle & style)
{
    render_->putArrows(nvidiaio::array_t(old_points), nvidiaio::array_t(new_points),
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

uint32_t RenderWrapper::getViewportWidth() const
{
    return render_->getViewportWidth();
}

uint32_t RenderWrapper::getViewportHeight() const
{
    return render_->getViewportHeight();
}

RenderWrapper::~RenderWrapper()
{
#ifndef __ANDROID__
    close();
#endif
}

} // namespace nvxio
