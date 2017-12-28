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

#ifndef EVENTPLAYER_HPP
#define EVENTPLAYER_HPP

#include <stdio.h>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "Render/RenderImpl.hpp"

namespace nvidiaio
{

class EventPlayer:
        public Render
{
public:
    EventPlayer():
        frameCounter(-1),
        loopCount(1),
        maxFrameIndex(-1),
        currentLoopIdx(0),
        keyBoardCallback(nullptr),
        keyboardCallbackContext(nullptr),
        mouseCallback(nullptr),
        mouseCallbackContext(nullptr)
    {
    }

    bool init(const std::string& path, int loops = 1);
    void final();

    void setEfficientRender(std::unique_ptr<Render> render);
    virtual ~EventPlayer();

    virtual void setOnKeyboardEventCallback(OnKeyboardEventCallback callback, void* context);
    virtual void setOnMouseEventCallback(OnMouseEventCallback callback, void* context);

    virtual void putImage(const image_t & image);
    virtual void putTextViewport(const std::string& text, const Render::TextBoxStyle& style);
    virtual void putFeatures(const array_t & location, const Render::FeatureStyle& style);
    virtual void putFeatures(const array_t & location, const array_t & styles);
    virtual void putLines(const array_t & lines, const Render::LineStyle& style);
    virtual void putConvexPolygon(const array_t & verticies, const LineStyle& style);
    virtual void putMotionField(const image_t & field, const Render::MotionFieldStyle& style);
    virtual void putObjectLocation(const nvxcu_rectangle_t & location, const Render::DetectedObjectStyle& style);
    virtual void putCircles(const array_t & circles, const CircleStyle& style);
    virtual void putArrows(const array_t & old_points, const array_t & new_points, const LineStyle& line_style);
    virtual bool flush();
    virtual void close();

    virtual uint32_t getViewportWidth() const
    {
        return efficientRender ? efficientRender->getViewportWidth() : 0u;
    }

    virtual uint32_t getViewportHeight() const
    {
        return efficientRender ? efficientRender->getViewportHeight() : 0u;
    }

protected:
    struct InputEvent
    {
        bool keyboard;
        uint32_t key;
        uint32_t x;
        uint32_t y;
    };

    bool readFrameEvents();
    void applyFrameEvents();

    std::unique_ptr<Render> efficientRender;
    std::ifstream logFile;
    std::string logLine;
    int32_t frameCounter;
    int32_t loopCount;
    int32_t maxFrameIndex;
    int32_t currentLoopIdx;

    std::vector<InputEvent> events;

    OnKeyboardEventCallback keyBoardCallback;
    void* keyboardCallbackContext;
    OnMouseEventCallback mouseCallback;
    void* mouseCallbackContext;
};

} // namespace nvidiaio

#endif // EVENTPLAYER_HPP
