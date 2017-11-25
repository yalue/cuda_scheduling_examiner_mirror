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

#ifndef EVENTLOGGER_HPP
#define EVENTLOGGER_HPP

#include <cstdio>
#include <memory>
#include <string>

#include "Render/RenderImpl.hpp"

namespace nvidiaio
{

class EventLogger:
        public Render
{
public:
    explicit EventLogger(bool _writeSrc);
    void setEfficientRender(std::unique_ptr<Render> render);
    bool init(const std::string& path);
    void final();
    virtual ~EventLogger();

    virtual void setOnKeyboardEventCallback(OnKeyboardEventCallback callback, void* context);
    virtual void setOnMouseEventCallback(OnMouseEventCallback callback, void* context);

    virtual void putImage(const image_t & image);
    virtual void putTextViewport(const std::string& text, const Render::TextBoxStyle& style);
    virtual void putFeatures(const array_t & location, const Render::FeatureStyle& style);
    virtual void putFeatures(const array_t & location, const array_t & styles);
    virtual void putLines(const array_t & lines, const Render::LineStyle& style);
    virtual void putConvexPolygon(const array_t & verticies, const LineStyle& style);
    virtual void putMotionField(const image_t & field, const Render::MotionFieldStyle& style);
    virtual void putObjectLocation(const nvxcu_rectangle_t& location, const Render::DetectedObjectStyle& style);
    virtual void putCircles(const array_t & circles, const CircleStyle& style);
    virtual void putArrows(const array_t & old_points, const array_t & new_points, const LineStyle& line_style);

    virtual bool flush();
    virtual void close();

    virtual uint32_t getViewportWidth() const
    {
        if (efficientRender)
        {
            return efficientRender->getViewportWidth();
        }
        else
        {
            return 0u;
        }
    }
    virtual uint32_t getViewportHeight() const
    {
        if (efficientRender)
        {
            return efficientRender->getViewportHeight();
        }
        else
        {
            return 0u;
        }
    }

protected:
    static void keyboard(void* context, char key, uint32_t x, uint32_t y);
    static void mouse(void* context, Render::MouseButtonEvent event, uint32_t x, uint32_t y);

    std::unique_ptr<Render> efficientRender;
    bool writeSrc;
    FILE* handle;
    std::string srcImageFilePattern;
    int frameCounter;
    OnKeyboardEventCallback keyBoardCallback;
    void* keyboardCallbackContext;
    OnMouseEventCallback mouseCallback;
    void* mouseCallbackContext;
};

} // namespace nvidiaio

#endif // EVENTLOGGER_HPP
