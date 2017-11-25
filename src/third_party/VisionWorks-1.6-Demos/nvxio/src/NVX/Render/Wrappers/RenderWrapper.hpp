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

#ifndef RENDER_WRAPPER_NVXCU_HPP
#define RENDER_WRAPPER_NVXCU_HPP

#include <memory>

#include <NVX/Render.hpp>

#ifdef __ANDROID__
# include "RenderImpl.hpp"
#else
# include "Render/RenderImpl.hpp"
#endif

namespace nvxio
{

class RenderWrapper :
        public Render
{
public:
    explicit RenderWrapper(std::unique_ptr<nvidiaio::Render> render);

#ifndef __ANDROID__
    virtual void setOnKeyboardEventCallback(OnKeyboardEventCallback callback, void * context);
    virtual void setOnMouseEventCallback(OnMouseEventCallback callback, void * context);
#endif

    virtual void putImage(const nvxcu_pitch_linear_image_t & image);
    virtual void putTextViewport(const std::string & text, const TextBoxStyle & style);
    virtual void putFeatures(const nvxcu_plain_array_t & location, const FeatureStyle & style);
    virtual void putFeatures(const nvxcu_plain_array_t &  location, const nvxcu_plain_array_t &  styles);
    virtual void putLines(const nvxcu_plain_array_t &  lines, const LineStyle & style);
    virtual void putConvexPolygon(const nvxcu_plain_array_t &  vertices, const LineStyle & style);
    virtual void putMotionField(const nvxcu_pitch_linear_image_t & field, const MotionFieldStyle & style);
    virtual void putObjectLocation(const nvxcu_rectangle_t & location, const DetectedObjectStyle & style);
    virtual void putCircles(const nvxcu_plain_array_t &  circles, const CircleStyle & style);
    virtual void putArrows(const nvxcu_plain_array_t &  old_points, const nvxcu_plain_array_t &  new_points, const LineStyle & style);

#ifndef __ANDROID__
    virtual bool flush();
    virtual void close();
#endif

    virtual uint32_t getViewportWidth() const;
    virtual uint32_t getViewportHeight() const;

    virtual ~RenderWrapper();

private:
    std::unique_ptr<nvidiaio::Render> render_;
};

} // namespace nvxio

#endif // RENDER_WRAPPER_NVXCU_HPP
