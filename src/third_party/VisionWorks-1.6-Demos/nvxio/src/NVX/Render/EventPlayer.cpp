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

#include <algorithm>

#include "Render/EventPlayer.hpp"

namespace nvidiaio
{

bool EventPlayer::init(const std::string &path, int loops)
{
    if (logFile.is_open())
    {
        // some log has been already opened
        return false;
    }

    if (loops < 1)
        return false;

    loopCount = loops;
    maxFrameIndex = -1;
    currentLoopIdx = 0;

    logFile.open(path);
    frameCounter = 0;

    return logFile.is_open();
}

// NOTE: flush method must be called before setEfficientRender
// It force EventPlayer to read all events for current frame and apply them
void EventPlayer::setEfficientRender(std::unique_ptr<Render> render)
{
    efficientRender = std::move(render);
}

bool EventPlayer::readFrameEvents()
{
    events.clear();
    int32_t frameIdx;
    InputEvent event;
    char tmp[256];

    if (currentLoopIdx >= loopCount)
    {
        return true;
    }

    do
    {
        if (!logLine.empty())
        {
            int status = sscanf(logLine.c_str(), "%d: %255s (%d,%d,%d)\n", &frameIdx, tmp, &event.key, &event.x, &event.y);
            if (status != 5)
            {
                // something strange
                continue;
            }
            maxFrameIndex = std::max(maxFrameIndex, frameIdx);
            if ((currentLoopIdx*maxFrameIndex + frameIdx) != frameCounter)
                return true;
            else
            {
                if (strcmp(tmp, "keyboard") == 0)
                {
                    event.keyboard = true;
                    events.push_back(event);
                }
                else if (strcmp(tmp, "mouse") == 0)
                {
                    event.keyboard = false;
                    events.push_back(event);
                }
                else
                {
                    // something strange
                    continue;
                }
            }
        }
    } while (std::getline(logFile, logLine));

    return false;
}

void EventPlayer::applyFrameEvents()
{
    for (size_t i = 0; i < events.size(); i++)
    {
        if (events[i].keyboard)
        {
            if (keyBoardCallback)
            {
                if (events[i].key != 27) // skip program finalization event
                    keyBoardCallback(keyboardCallbackContext, events[i].key, events[i].x, events[i].y);
            }
        }
        else
        {
            if (mouseCallback)
                mouseCallback(mouseCallbackContext, (MouseButtonEvent)events[i].key, events[i].x, events[i].y);
        }
    }
}

void EventPlayer::final()
{
    logFile.close();

    frameCounter = -1;
}

EventPlayer::~EventPlayer()
{
    final();
}

void EventPlayer::setOnKeyboardEventCallback(OnKeyboardEventCallback callback, void *context)
{
    keyBoardCallback = callback;
    keyboardCallbackContext = context;
}

void EventPlayer::setOnMouseEventCallback(OnMouseEventCallback callback, void *context)
{
    mouseCallback = callback;
    mouseCallbackContext = context;
}

void EventPlayer::putTextViewport(const std::string &text, const TextBoxStyle &style)
{
    if (efficientRender)
        efficientRender->putTextViewport(text, style);
}

void EventPlayer::putImage(const image_t & image)
{
    if (efficientRender)
        efficientRender->putImage(image);
}

void EventPlayer::putObjectLocation(const nvxcu_rectangle_t &location, const DetectedObjectStyle &style)
{
    if (efficientRender)
        efficientRender->putObjectLocation(location, style);
}

void EventPlayer::putFeatures(const array_t & location, const FeatureStyle &style)
{
    if (efficientRender)
        efficientRender->putFeatures(location, style);
}

void EventPlayer::putFeatures(const array_t & location, const array_t & styles)
{
    if (efficientRender)
        efficientRender->putFeatures(location, styles);
}

void EventPlayer::putLines(const array_t & lines, const Render::LineStyle &style)
{
    if (efficientRender)
        efficientRender->putLines(lines, style);
}

void EventPlayer::putConvexPolygon(const array_t & verticies, const LineStyle& style)
{
    if (efficientRender)
        efficientRender->putConvexPolygon(verticies, style);
}

void EventPlayer::putMotionField(const image_t & field, const MotionFieldStyle &style)
{
    if (efficientRender)
        efficientRender->putMotionField(field, style);
}

void EventPlayer::putCircles(const array_t & circles, const CircleStyle& style)
{
    if (efficientRender)
        efficientRender->putCircles(circles, style);
}

void EventPlayer::putArrows(const array_t & old_points, const array_t & new_points, const LineStyle& line_style)
{
    if (efficientRender)
        efficientRender->putArrows(old_points, new_points, line_style);
}

bool EventPlayer::flush()
{
    bool io_error = false;

    if (!readFrameEvents())
    {
        currentLoopIdx++;
        if (currentLoopIdx < loopCount)
        {
            logFile.clear();
            io_error = !logFile.seekg(0, std::ios_base::beg);
        }
    }

    ++frameCounter;

    bool status = true;
    if (efficientRender)
        status = efficientRender->flush();

    if (status)
        applyFrameEvents();

    if (io_error || currentLoopIdx >= loopCount)
    {
        // Exit by <ESC> button if scenario looping is enabled
        if (keyBoardCallback)
            keyBoardCallback(keyboardCallbackContext, 27, 0, 0);
    }

    return status;
}

void EventPlayer::close()
{
    frameCounter = -1;

    if (efficientRender)
        efficientRender->close();
}

} // namespace nvidiaio
