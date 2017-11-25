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

#ifndef NVXCUIO_RENDER_HPP
#define NVXCUIO_RENDER_HPP

#include <memory>
#include <string>

#include <NVX/nvxcu.h>
#include <NVX/Export.hpp>

#ifndef __ANDROID__
# include <NVX/Application.hpp>
#endif

/**
 * \file
 * \brief The VisionWorks CUDA `Render` interface and utility functions.
 */

namespace nvxio
{

/**
 * \defgroup group_nvxcu_render VisionWorks CUDA Render Interface
 * \ingroup group_nvxio_render
 *
 * This class is designed to be VisionWorks CUDA interface for rendering 2D images and some primitive 2D graphic
 * objects, like lines, circles, text boxes,
 * @if NVX_DOCS_ANDROID
 * on the screen.
 * @else
 * on the screen, to video file or single image.
 * @endif
 */

/**
 * \ingroup group_nvxcu_render
 * \brief `%Render` interface.
 *
 * \see nvxio::Render
 */
class NVXIO_EXPORT Render
{
public:

    /**
     * \brief Defines the Render types.
     */
    enum TargetType
    {
        UNKNOWN_RENDER, /**< \brief Indicates a stub render. */
        WINDOW_RENDER, /**< \brief Indicates a window render. */
        VIDEO_RENDER, /**< \brief Indicates a render for video writing. */
        IMAGE_RENDER /**< \brief Indicates a render for image writing. */
    };

    /**
     * \brief Defines the text box parameters.
     */
    struct TextBoxStyle
    {
        uint8_t color[4]; /**< \brief Holds the text color in RGBA format. */
        uint8_t bgcolor[4]; /**< \brief Holds the background color of the box. */
        nvxcu_coordinates2d_t origin; /**< \brief Holds the coordinates of the top-left corner of the box. */
    };

    /**
     * \brief Defines the features parameters.
     */
    struct FeatureStyle
    {
        uint8_t color[4]; /**< \brief Holds the feature color in RGBA format. */
        float radius; /**< \brief Holds the radius of the feature. */
    };

    /**
     * \brief Defines line style.
     */
    struct LineStyle
    {
        uint8_t color[4]; /**< \brief Holds the line color in RGBA format. */
        int32_t thickness; /**< \brief Holds the thickness of the line. */
    };

    /**
     * \brief Defines motion field style.
     */
    struct MotionFieldStyle
    {
        uint8_t color[4]; /**< \brief Holds the color of the motion field in RGBA format. */
    };


    /**
     * \brief Defines the detected object's rectangle style.
     */
    struct DetectedObjectStyle
    {
        /** \brief Holds the text label. */
        std::string label;
        /** \brief Holds the line color in RGBA format. */
        uint8_t color[4];
        /** \brief Holds the line thickness. */
        uint8_t thickness;
        /** \brief Holds the radius of the corners. */
        uint8_t radius;
        /** \brief Holds a flag indicating whether the detected object should be filled with half-transparent color. */
        bool isHalfTransparent;
    };

    /**
     * \brief Defines circle style.
     */
    struct CircleStyle
    {
        uint8_t color[4]; /**< \brief Holds the line color in RGBA format. */
        int32_t thickness; /**< \brief Holds the line thickness. */
    };

#ifndef __ANDROID__
    /**
     * \brief Defines mouse events.
     */
    enum MouseButtonEvent
    {
        LeftButtonDown, /**< \brief Indicates the left mouse button has been pressed down. */
        LeftButtonUp, /**< \brief Indicates the left mouse button has been released. */
        MiddleButtonDown, /**< \brief Indicates a middle mouse button has been pressed down. */
        MiddleButtonUp, /**< \brief Indicates the middle mouse button has been released. */
        RightButtonDown, /**< \brief Indicates the right mouse button has been pressed down. */
        RightButtonUp, /**< \brief Indicates the right mouse button has been released. */
        MouseMove /**< \brief Indicates the mouse has been moved. */
    };

    /**
     * \brief Callback for keyboard events.
     * \param [in] context A pointer to data to be passed to the callback.
     * \param [in] key Specifies the keyboard key that corresponds to the event.
     * \param [in] x Specifies the x-coordinate of the mouse position.
     * \param [in] y Specifies the y-coordinate of the mouse position.
     */
    typedef void (*OnKeyboardEventCallback)(void* context, char key, uint32_t x, uint32_t y);

    /**
     * \brief Callback for mouse events.
     * \param [in] context A pointer to data to be passed to the callback.
     * \param [in] event Specifies the mouse event.
     * \param [in] x Specifies the x-coordinate of the mouse position.
     * \param [in] y Specifies the y-coordinate of the mouse position.
     */
    typedef void (*OnMouseEventCallback)(void* context, MouseButtonEvent event, uint32_t x, uint32_t y);

    /**
     * \brief Sets the keyboard event callback.
     * \param [in] callback Specifies the callback to set.
     * \param [in] context A pointer to data to be passed to the callback.
     */
    virtual void setOnKeyboardEventCallback(OnKeyboardEventCallback callback, void* context) = 0;

    /**
     * \brief Sets mouse event callback.
     * \param [in] callback Callback.
     * \param [in] context A pointer to data to be passed to the callback.
     */
    virtual void setOnMouseEventCallback(OnMouseEventCallback callback, void* context) = 0;
#endif

    /**
     * \brief Puts the image to the render.
     * \param [in] image Specifies the image.
     */
    virtual void putImage(const nvxcu_pitch_linear_image_t & image) = 0;

    /**
     * \brief Puts a message box on the viewport.
     * \param [in] text A reference to the text of the message.
     * \param [in] style A reference to the style of the message box.
     */
    virtual void putTextViewport(const std::string& text, const TextBoxStyle& style) = 0;

    /**
     * \brief Puts features on the image.
     * \param [in] location Specifies an array of \ref nvxcu_keypoint_t, \ref nvxcu_point2f_t or \ref nvxcu_keypointf_t structures.
     * \param [in] style A reference to the style for the features.
     */
    virtual void putFeatures(const nvxcu_plain_array_t & location, const FeatureStyle& style) = 0;

    /**
     * \brief Puts features on the image.
     * \param [in] location Specifies an array of \ref nvxcu_keypoint_t, \ref nvxcu_point2f_t or \ref nvxcu_keypointf_t structures.
     * \param [in] styles A reference to the array of \ref Render::FeatureStyle for the features.
     *
     * \par Example Code
     * @snippet nvxio.cpp nvxcuio_render_put_features_with_style
     */
    virtual void putFeatures(const nvxcu_plain_array_t & location, const nvxcu_plain_array_t & styles) = 0;

    /**
     * \brief Puts lines on the image.
     * \param [in] lines Specifies an array of lines. Each line is encoded as \ref nvxcu_point4f_t (x1, y1, x2, y2).
     * \param [in] style A reference to the style of the lines.
     */
    virtual void putLines(const nvxcu_plain_array_t & lines, const LineStyle& style) = 0;

    /**
     * \brief Puts a convex polygon on the image.
     * \param [in] vertices Specifies an array of polygon's vertices (\ref nvxcu_coordinates2d_t).
     * \param [in] style A reference to the style of the polygon.
     */
    virtual void putConvexPolygon(const nvxcu_plain_array_t & vertices, const LineStyle& style) = 0;

    /**
     * \brief Puts motion field on the image.
     * \param [in] field Specfies a 2-channel image (\ref NVXCU_DF_IMAGE_2F32), each pixel corresponding to vector of motion.
     * \param [in] style A reference to the style of the motion field.
     */
    virtual void putMotionField(const nvxcu_pitch_linear_image_t & field, const MotionFieldStyle& style) = 0;

    /**
     * \brief Puts object location on the image.
     * \param [in] location A reference to the rectangle.
     * \param [in] style A reference to the style of the object location.
     */
    virtual void putObjectLocation(const nvxcu_rectangle_t & location, const DetectedObjectStyle& style) = 0;

    /**
     * \brief Puts circles on the image.
     * \param [in] circles Specifies an array of circles. Each circle is encoded as a \ref nvxcu_point3f_t (x, y, radius).
     * \param [in] style A reference to the style of the object location.
     */
    virtual void putCircles(const nvxcu_plain_array_t & circles, const CircleStyle& style) = 0;

    /**
     * \brief Puts arrows on the image.
     * \param [in] old_points     Specifies an array of arrow start points.
     * \param [in] new_points     Specifies an array of arrow end points.
     * \param [in] style          Specifies the style of the arrows's lines.
     */
    virtual void putArrows(const nvxcu_plain_array_t & old_points, const nvxcu_plain_array_t & new_points,
                           const LineStyle& style) = 0;

#ifndef __ANDROID__
    /**
     * \brief Renders all primitives.
     * \return Status of the operation. Returns `true` if rendering procedure is successful;
     * returns `false` if render is not initialized properly or has been closed by the user.
     */
    virtual bool flush() = 0;

    /**
     * \brief Closes the render.
     */
    virtual void close() = 0;
#endif

    /**
     * \brief Gets the target type \see TargetType.
     * \return Target type.
     */
    TargetType getTargetType() const
    {
        return targetType;
    }

    /**
     * \brief Gets the render name.
     * \return Render name.
     */
    std::string getRenderName() const
    {
        return renderName;
    }

    /**
     * \brief Gets the viewport width.
     * \return Width.
     */
    virtual uint32_t getViewportWidth() const = 0;

    /**
     * \brief Gets the viewport height.
     * \return Height.
     */
    virtual uint32_t getViewportHeight() const = 0;

    /**
     * \brief Destructor.
     */
    virtual ~Render()
    {}

protected:

    Render(TargetType type = Render::UNKNOWN_RENDER, std::string name = "Undefined"):
        targetType(type),
        renderName(name)
    {}

    const TargetType  targetType;
    const std::string renderName;
};

#ifdef __ANDROID__

/**
 * \ingroup group_nvxcu_render
 * \brief Render factory that creates UI render with a window by default.
 * \param [in] width Specifies the width of the render.
 * \param [in] height Specifies the height of the render.
 * \param [in] doScale Specifies whether render should scale image to fit the window (used only in `window` mode).
 *
 * \return `%Render` implementation or `nullptr` if this one cannot be created.
 * \see nvxio::createRender
 */
NVXIO_EXPORT std::unique_ptr<Render> createRender(uint32_t width, uint32_t height, bool doScale = true);

#else

/**
 * \ingroup group_nvxcu_render
 * \brief Render factory that creates UI render with a window by default.
 * \param [in] title A reference to the title of the render.
 * \param [in] width Specifies the width of the render.
 * \param [in] height Specifies the height of the render.
 * \param [in] format Specifies the format of the render.
 * \param [in] doScale Specifies whether render should scale the image to fit the window (used only in `window` mode).
 * \param [in] fullScreen Specifies whether render should be fullscreen.
 *
 * \return `%Render` implementation or `nullptr` if this one cannot be created.
 * \see nvxio::createDefaultRender
 */
NVXIO_EXPORT std::unique_ptr<Render> createDefaultRender(const std::string& title, uint32_t width, uint32_t height,
                                                         nvxcu_df_image_e format = NVXCU_DF_IMAGE_RGBX, bool doScale = true,
                                                         bool fullScreen = nvxio::Application::get().getFullScreenFlag());

/**
 * \ingroup group_nvxcu_render
 * \brief Creates a render for writing video.
 *
 * Use `NVXIO_VIDEO_RENDER_BITRATE` environment variable to specify the target video bitrate in bit/s.
 * \param [in] path A references to the path to output video file.
 * \param [in] width Specifies the width of the render.
 * \param [in] height Specifies the height of the render.
 * \param [in] format Specifies the format of the render.
 *
 * \return `%Render` implementation or `nullptr` if this one cannot be created.
 * \see nvxio::createVideoRender
 */
NVXIO_EXPORT std::unique_ptr<Render> createVideoRender(const std::string& path, uint32_t width,
                                                       uint32_t height, nvxcu_df_image_e format = NVXCU_DF_IMAGE_RGBX);

/**
 * \ingroup group_nvxcu_render
 * \brief Creates a Window render.
 *
 * You can use `NVXIO_DISPLAY` environment variable to specify display to use in full screen mode.
 * \param [in] title A reference to the title of the window.
 * \param [in] width Specifies the width of the render.
 * \param [in] height Specifies the height of the render.
 * \param [in] format Specifies the format of the render.
 * \param [in] doScale Whether render should scale image to fit the window (used only in `window` mode).
 * \param [in] fullscreen Whether render should be fullscreen.
 *
 * \return `%Render` implementation or `nullptr` if this one cannot be created.
 * \see nvxio::createWindowRender
 */
NVXIO_EXPORT std::unique_ptr<Render> createWindowRender(const std::string& title, uint32_t width, uint32_t height,
                                                        nvxcu_df_image_e format = NVXCU_DF_IMAGE_RGBX, bool doScale = true,
                                                        bool fullscreen = nvxio::Application::get().getFullScreenFlag());

/**
 * \ingroup group_nvxcu_render
 * \brief Creates a render for image sequence.
 * \param [in] path A reference to the output image sequence path.
 * \param [in] width Specifies the width of the render.
 * \param [in] height Specifies the height of the render.
 * \param [in] format Specifies the format of the render.
 *
 * \return `%Render` implementation or `nullptr` if this one cannot be created.
 * \see nvxio::createImageRender
 */
NVXIO_EXPORT std::unique_ptr<Render> createImageRender(const std::string& path, uint32_t width, uint32_t height,
                                                       nvxcu_df_image_e format = NVXCU_DF_IMAGE_RGBX);

#endif

} // namespace nvxio

#endif // NVXCUIO_RENDER_HPP
