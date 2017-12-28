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


#include "Render/EventLogger.hpp"

#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

#include <cuda_runtime_api.h>

#ifdef USE_OPENCV
# include <opencv2/imgproc/imgproc.hpp>
# include <opencv2/highgui/highgui.hpp>
#endif

static bool ComparatorPoint3f (const nvxcu_point3f_t & a, const nvxcu_point3f_t & b)
{
    if (a.x == b.x)
    {
        if (a.y == b.y)
            return a.z < b.z;
        else
            return a.y < b.y;
    }
    else
        return a.x < b.x;
}

static bool ComparatorPoint4f (const nvxcu_point4f_t & a, const nvxcu_point4f_t & b)
{
    if (a.x == b.x)
    {
        if (a.y == b.y)
        {
            if (a.z == b.z)
                return a.w < b.w;
            else
                return a.z < b.z;
        }
        else
            return a.y < b.y;
    }
    else
        return a.x < b.x;
}

namespace nvidiaio
{

EventLogger::EventLogger(bool _writeSrc):
    writeSrc(_writeSrc),
    handle(nullptr),
    frameCounter(-1),
    keyBoardCallback(nullptr),
    mouseCallback(nullptr)
{
}

bool EventLogger::init(const std::string &path)
{
    if (handle)
    {
        // some log has been already opened
        return true;
    }

    size_t dot = path.find_last_of('.');
    std::string baseName = path.substr(0, dot);
    std::string ext = path.substr(dot, std::string::npos);

    handle = fopen(path.c_str(), "rt");
    if (handle)
    {
        // file with this name already exists that means that render was reopened
        int logNameIdx = 0;
        do
        {
            fclose(handle);
            logNameIdx++;
            handle = fopen((baseName+std::to_string(logNameIdx)+ext).c_str(), "rt");
        }
        while (handle);

        srcImageFilePattern = baseName + std::to_string(logNameIdx) + "_src_%05d.png";
        handle = fopen((baseName+std::to_string(logNameIdx)+ext).c_str(), "wt");
    }
    else
    {
        srcImageFilePattern = baseName + "_src_%05d.png";
        handle = fopen(path.c_str(), "wt");
    }

    frameCounter = 0;

    return handle != nullptr;
}

void EventLogger::setEfficientRender(std::unique_ptr<Render> render)
{
    efficientRender = std::move(render);
    if (efficientRender)
    {
        efficientRender->setOnKeyboardEventCallback(keyboard, this);
        efficientRender->setOnMouseEventCallback(mouse, this);
    }
}

void EventLogger::final()
{
    if (handle)
        fclose(handle);

    frameCounter = -1;
}

EventLogger::~EventLogger()
{
    final();
}

void EventLogger::keyboard(void* context, char key, uint32_t x, uint32_t y)
{
    EventLogger* self = (EventLogger*)context;
    if (!context)
        return;
    if (self->handle)
        fprintf(self->handle, "%d: keyboard (%d,%u,%u)\n", self->frameCounter, key, x, y);

    if (self->keyBoardCallback)
        self->keyBoardCallback(self->keyboardCallbackContext, key, x, y);
}

void EventLogger::mouse(void* context, Render::MouseButtonEvent event, uint32_t x, uint32_t y)
{
    EventLogger* self = (EventLogger*)context;
    if (!context)
        return;

    if (self->handle)
        fprintf(self->handle, "%d: mouse (%d,%u,%u)\n", self->frameCounter, (int)event, x, y);

    if (self->mouseCallback)
        self->mouseCallback(self->mouseCallbackContext, event, x, y);
}

void EventLogger::putTextViewport(const std::string &text, const Render::TextBoxStyle &style)
{
    if (handle)
    {
        std::string filtered = "";
        size_t curr_pos = 0;
        size_t prev_pos = 0;
        while((curr_pos = text.find("\n", prev_pos)) != std::string::npos)
        {
            filtered += text.substr(prev_pos, curr_pos-prev_pos);
            filtered += "\\n";
            prev_pos = curr_pos+1;
        }

        filtered += text.substr(prev_pos, std::string::npos);

        fprintf(handle, "%d: textBox(color(%u,%u,%u,%u), bkcolor(%u,%u,%u,%u), origin(%u,%u), \"%s\")\n",
                frameCounter,
                style.color[0], style.color[1], style.color[2], style.color[3],
                style.bgcolor[0], style.bgcolor[1], style.bgcolor[2], style.bgcolor[3],
                style.origin.x, style.origin.y,
                filtered.c_str()
               );
    }

    if (efficientRender)
        efficientRender->putTextViewport(text, style);
}

void EventLogger::putImage(const image_t & image)
{
    if (handle)
    {
        nvxcu_df_image_e format = image.format;
        fprintf(handle, "%d: image(%d, %dx%d)\n",
                frameCounter, format, image.width, image.height);

#ifdef USE_OPENCV
        if (writeSrc)
        {
            int matType = format == NVXCU_DF_IMAGE_RGBX ? CV_8UC4 :
                          format == NVXCU_DF_IMAGE_RGB ? CV_8UC3 :
                          format == NVXCU_DF_IMAGE_U8 ? CV_8UC1: -1;

            if (matType < 0)
            {
                char sFormat[sizeof(format)+1];
                std::memcpy(sFormat, &format, sizeof(format));
                sFormat[sizeof(format)] = '\0';

                NVXIO_THROW_EXCEPTION( "Dumping frames in format " << sFormat << " is not supported" );
                return;
            }

            {
                Image2CPUPointerMapper mapper(image);

                cv::Mat srcFrame(image.height, image.width, matType, (void *)(const void *)mapper);
                cv::Mat normalizedFrame;

                if (format == NVXCU_DF_IMAGE_U8)
                    normalizedFrame = srcFrame;
                else
                {
                    cv::cvtColor(srcFrame, normalizedFrame,
                                 format == NVXCU_DF_IMAGE_RGBX ? CV_RGBA2BGRA : CV_RGB2BGR);
                }

                std::string name = cv::format(srcImageFilePattern.c_str(), frameCounter);

                if (!cv::imwrite(name, normalizedFrame))
                    fprintf(stderr, "Cannot write frame to %s\n", name.c_str());
            }
        }
#endif // USE_OPENCV
    }

    if (efficientRender)
        efficientRender->putImage(image);
}

void EventLogger::putObjectLocation(const nvxcu_rectangle_t &location, const Render::DetectedObjectStyle &style)
{
    if (handle)
    {
        fprintf(handle, "%d: object(color(%u,%u,%u,%u), location(%u,%u,%u,%u), \"%s\")\n",
                frameCounter,
                style.color[0], style.color[1], style.color[2], style.color[3],
                location.start_x, location.start_y, location.end_x, location.end_y,
                style.label.c_str()
                );
    }

    if (efficientRender)
        efficientRender->putObjectLocation(location, style);
}

void EventLogger::putFeatures(const array_t & location, const Render::FeatureStyle &style)
{
    if (handle)
    {
        nvxcu_array_item_type_e item_type = location.item_type;
        NVXIO_ASSERT( (item_type == NVXCU_TYPE_KEYPOINT) || (item_type == NVXCU_TYPE_POINT2F) || (item_type == NVXCU_TYPE_KEYPOINTF) );

        fprintf(handle, "%d: features(color(%u,%u,%u,%u), %u",
                frameCounter,
                style.color[0], style.color[1], style.color[2], style.color[3],
                location.num_items);

        if (location.num_items > 0u)
        {
            Array2CPUPointerMapper mapper(location);

            if (item_type == NVXCU_TYPE_KEYPOINT)
            {
                const nvxcu_keypoint_t * featureData = static_cast<const nvxcu_keypoint_t *>(mapper);

                for (uint32_t i = 0u; i < location.num_items; i++)
                {
                    nvxcu_keypoint_t feature = featureData[i];
                    fprintf(handle, ",ftr(%d,%d)", feature.x, feature.y);
                }
            }
            else if (item_type == NVXCU_TYPE_POINT2F)
            {
                const nvxcu_point2f_t * featureData = static_cast<const nvxcu_point2f_t *>(mapper);

                for (uint32_t i = 0u; i < location.num_items; i++)
                {
                    nvxcu_point2f_t feature = featureData[i];
                    fprintf(handle, ",ftr(%.1f,%.1f)", feature.x, feature.y);
                }
            }
            else if (item_type == NVXCU_TYPE_KEYPOINTF)
            {
                const nvxcu_keypointf_t * featureData = static_cast<const nvxcu_keypointf_t *>(mapper);

                for (uint32_t i = 0u; i < location.num_items; i++)
                {
                    nvxcu_keypointf_t feature = featureData[i];
                    fprintf(handle, ",ftr(%.1f,%.1f)", feature.x, feature.y);
                }
            }
        }

        fprintf(handle, ")\n");
    }

    if (efficientRender)
        efficientRender->putFeatures(location, style);
}

void EventLogger::putFeatures(const array_t & location, const array_t & styles)
{
    if (handle)
    {
        nvxcu_array_item_type_e item_type = location.item_type;
        NVXIO_ASSERT( (item_type == NVXCU_TYPE_KEYPOINT) || (item_type == NVXCU_TYPE_POINT2F) || (item_type == NVXCU_TYPE_KEYPOINTF) );
        NVXIO_ASSERT( location.num_items == styles.num_items );

        fprintf(handle, "%d: features(%u", frameCounter, location.num_items);

        if (location.num_items > 0u)
        {
            Array2CPUPointerMapper styleMapper(styles), locationMapper(location);

            const Render::FeatureStyle * styleData = static_cast<const Render::FeatureStyle *>(styleMapper);

            if (item_type == NVXCU_TYPE_KEYPOINT)
            {
                const nvxcu_keypoint_t * featureData = static_cast<const nvxcu_keypoint_t *>(locationMapper);

                for (uint32_t i = 0u; i < location.num_items; i++)
                {
                    const nvxcu_keypoint_t & feature = featureData[i];
                    const Render::FeatureStyle & style = styleData[i];

                    fprintf(handle, ",ftr(%d,%d,%u,%u,%u,%u)", feature.x, feature.y,
                            style.color[0], style.color[1], style.color[2], style.color[3]);
                }
            }
            else if (item_type == NVXCU_TYPE_POINT2F)
            {
                const nvxcu_point2f_t * featureData = static_cast<const nvxcu_point2f_t *>(locationMapper);

                for (uint32_t i = 0u; i < location.num_items; i++)
                {
                    const nvxcu_point2f_t & feature = featureData[i];
                    const Render::FeatureStyle & style = styleData[i];

                    fprintf(handle, ",ftr(%.1f,%.1f,%u,%u,%u,%u)", feature.x, feature.y,
                            style.color[0], style.color[1], style.color[2], style.color[3]);
                }
            }
            else if (item_type == NVXCU_TYPE_KEYPOINTF)
            {
                const nvxcu_keypointf_t * featureData = static_cast<const nvxcu_keypointf_t *>(locationMapper);

                for (uint32_t i = 0u; i < location.num_items; i++)
                {
                    const nvxcu_keypointf_t & feature = featureData[i];
                    const Render::FeatureStyle & style = styleData[i];

                    fprintf(handle, ",ftr(%.1f,%.1f,%u,%u,%u,%u)", feature.x, feature.y,
                            style.color[0], style.color[1], style.color[2], style.color[3]);
                }
            }
        }

        fprintf(handle, ")\n");
    }

    if (efficientRender)
        efficientRender->putFeatures(location, styles);
}

void EventLogger::putLines(const array_t & lines, const Render::LineStyle &style)
{
    if (handle)
    {
        fprintf(handle, "%d: lines(color(%u,%u,%u,%u), thickness(%d), %u",
                frameCounter,
                style.color[0], style.color[1], style.color[2], style.color[3],
                style.thickness,
                lines.num_items);

        if (lines.num_items > 0u)
        {
            Array2CPUPointerMapper mapper(lines);

            nvxcu_point4f_t * linesData = (nvxcu_point4f_t *)(const nvxcu_point4f_t *)(mapper);

            std::sort(linesData, linesData + lines.num_items, &ComparatorPoint4f);

            for (uint32_t i = 0u; i < lines.num_items; i++)
            {
                fprintf(handle, ",line(%d,%d,%d,%d)",
                        static_cast<int32_t>(linesData[i].x),
                        static_cast<int32_t>(linesData[i].y),
                        static_cast<int32_t>(linesData[i].z),
                        static_cast<int32_t>(linesData[i].w));
            }
        }

        fprintf(handle, ")\n");
    }

    if (efficientRender)
        efficientRender->putLines(lines, style);
}

void EventLogger::putConvexPolygon(const array_t & verticies, const LineStyle& style)
{
    if (handle)
    {
        fprintf(handle, "%d: polygon(color(%u,%u,%u,%u), thickness(%d), %u",
                frameCounter,
                style.color[0], style.color[1], style.color[2], style.color[3],
                style.thickness,
                verticies.num_items);

        if (verticies.num_items > 0u)
        {
            Array2CPUPointerMapper mapper(verticies);

            const nvxcu_coordinates2d_t * verticiesData = static_cast<const nvxcu_coordinates2d_t *>(mapper);

            for (uint32_t i = 0u; i < verticies.num_items; i++)
            {
                const nvxcu_coordinates2d_t & item = verticiesData[i];
                fprintf(handle, ",vertex(%u,%u)", item.x, item.y);
            }
        }

        fprintf(handle, ")\n");
    }

    if (efficientRender)
        efficientRender->putConvexPolygon(verticies, style);
}

void EventLogger::putMotionField(const image_t & field, const Render::MotionFieldStyle &style)
{
    if (handle)
        fprintf(handle, "%d: motionField(color(%u,%u,%u,%u)",
                frameCounter, style.color[0], style.color[1], style.color[2], style.color[3]);

    fprintf(handle, ",%dx%d", field.width, field.height);

    {
        Image2CPUPointerMapper mapper(field);

        const float * fieldData = static_cast<const float *>(mapper);
        uint32_t pitch = field.width << 1;

        for (uint32_t y = 0u; y < field.height; y++)
        {
            const float * fieldRow = fieldData + pitch * y;

            for (uint32_t x = 0u; x < pitch; x += 2)
                fprintf(handle, ",%f,%f", fieldRow[x], fieldRow[x + 1]);
        }
    }

    fprintf(handle, ")\n");

    if (efficientRender)
        efficientRender->putMotionField(field, style);
}

void EventLogger::putCircles(const array_t & circles, const CircleStyle& style)
{
    if (handle)
    {
        fprintf(handle, "%d: circles(color(%u,%u,%u,%u), thickness(%d), %u",
                frameCounter,
                style.color[0], style.color[1], style.color[2], style.color[3],
                style.thickness,
                circles.num_items);

        if (circles.num_items > 0u)
        {
            Array2CPUPointerMapper mapper(circles);

            nvxcu_point3f_t * circlesData = (nvxcu_point3f_t *)(const nvxcu_point3f_t *)(mapper);

            std::sort(circlesData, circlesData + circles.num_items, &ComparatorPoint3f);

            for (uint32_t i = 0u; i < circles.num_items; i++)
            {
                const nvxcu_point3f_t & circle = circlesData[i];
                fprintf(handle, ",circle(%f,%f,%f)", circle.x, circle.y, circle.z);
            }
        }

        fprintf(handle, ")\n");
    }

    if (efficientRender)
        efficientRender->putCircles(circles, style);
}

void EventLogger::putArrows(const array_t & old_points, const array_t & new_points,
                            const LineStyle& line_style)
{
    if (handle)
    {
        uint32_t num_items = std::min(old_points.num_items, new_points.num_items);

        fprintf(handle, "%d: arrows(color(%u,%u,%u,%u), thickness(%d), %u)\n",
                frameCounter,
                line_style.color[0], line_style.color[1], line_style.color[2], line_style.color[3],
                line_style.thickness,
                num_items);
    }

    if (efficientRender)
        efficientRender->putArrows(old_points, new_points, line_style);
}

bool EventLogger::flush()
{
    ++frameCounter;

    if (handle)
        fflush(handle);

    if (efficientRender)
        return efficientRender->flush();
    else
        return true;
}

void EventLogger::close()
{
    frameCounter = -1;

    if (efficientRender)
        efficientRender->close();
}

void EventLogger::setOnKeyboardEventCallback(OnKeyboardEventCallback callback, void *context)
{
    keyBoardCallback = callback;
    keyboardCallbackContext = context;
}

void EventLogger::setOnMouseEventCallback(OnMouseEventCallback callback, void *context)
{
    mouseCallback = callback;
    mouseCallbackContext = context;
}

}
