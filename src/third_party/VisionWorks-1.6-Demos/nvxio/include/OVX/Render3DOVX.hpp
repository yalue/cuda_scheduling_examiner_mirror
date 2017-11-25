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

#ifndef NVXIO_RENDER3D_HPP
#define NVXIO_RENDER3D_HPP

#include <string>
#include <vector>

#include <VX/vx.h>
#include <NVX/Export.hpp>
#include <OVX/RenderOVX.hpp>

/**
 * \file
 * \brief The `Render3D` interface and utility functions.
 */

namespace ovxio {

/**
 * \defgroup group_nvxio_render3d Render3D
 * \ingroup nvx_nvxio_api
 *
 * This class is intended to display primitives in the 3D space.
 * `%Render3D` implementation allows displaying images, using primitives (point clouds, planes), and handling keyboard/mouse events.
 */

/**
 * \defgroup group_nvx_render3d VisionWorks Render3D Interface
 * \ingroup group_nvxio_render3d
 *
 * This class is designed to be the VisionWorks interface to display primitives in the 3D space.
 */

/**
 * \ingroup group_nvx_render3d
 * \brief `%Render3D` interface.
 *
 * \see nvxio::Render3D
 */
class NVXIO_EXPORT Render3D
{
public:

    /**
     * \brief Defines `%Render3D` types.
     */
    enum TargetType
    {
        UNKNOWN_RENDER, /**< \brief Indicates a stub render. */
        BASE_RENDER_3D /**< \brief Indicates Render3D. */
    };

    /**
     * \brief Holds the plane style.
     *
     * This defines the color of the planes, which depend on the distance to the camera.
     * The color varies linearly with the distance from `red` to `green` (without the `blue` component).
     */
    struct PlaneStyle
    {
        vx_float32 minDistance; /**< \brief Holds the minimal distance that a plane can have. It corresponds to the `red` color. */
        vx_float32 maxDistance; /**< \brief Holds the maximal distance that a plane can have. It corresponds to the `green` color. */
    };

    /**
     * \brief Holds the point cloud style.
     *
     * This defines the colors of the points of the cloud, which depend on the distance to the camera.
     * The color varies linearly with the distance from `red` to `green` (without the `blue` component).
     */
    struct PointCloudStyle
    {
        vx_float32 minDistance; /**< \brief Holds the minimal distance that a point can have. It corresponds to the `red` color. */
        vx_float32 maxDistance; /**< \brief Holds the maximal distance that a point can have. It corresponds to the `green` color. */
    };

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
    typedef void (*OnKeyboardEventCallback)(void* context, vx_char key, vx_uint32 x, vx_uint32 y);
    /**
     * \brief Callback for mouse events.
     * \param [in] context A pointer to data to be passed to the callback.
     * \param [in] event Specifies the mouse event.
     * \param [in] x Specifies the x-coordinate of the mouse position.
     * \param [in] y Specifies the y-coordinate of the mouse position.
     */
    typedef void (*OnMouseEventCallback)(void* context, MouseButtonEvent event, vx_uint32 x, vx_uint32 y);

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

    /**
     * \brief Puts the surface to the render.
     * The surface consists of 2 adjacent triangles.
     * \param [in] planes Array of surface. The surface is encoded as 4 \ref nvx_point3f_t points that define the vertices of the triangles.
     * If the points are (p1,p2,p3,p4), then (p1,p2,p3) define the first triangle and (p1,p3,p4) define the second.
     * \param [in] model Specifies the model matrix that transforms Model Space to World Space coordinates.
     * \param [in] style A reference to the style of the surface.
     */
    virtual void putPlanes(vx_array planes, vx_matrix model, const PlaneStyle& style) = 0;

    /**
     * \brief Puts the point cloud to the render.
     * \param [in] points Array of the points. The point is encoded as \ref nvx_point3f_t.
     * \param [in] model Specifies the model matrix that transforms Model Space to World Space coordinates.
     * \param [in] style A reference to the style of the point cloud.
     */
    virtual void putPointCloud(vx_array points, vx_matrix model, const PointCloudStyle& style) = 0;


    /**
     * \brief Renders all primitives.
     * \return Status of the operation.
     */
    virtual bool flush() = 0;

    /**
     * \brief Closes the render.
     */
    virtual void close() = 0;

    /**
     * \brief Sets view matrix.
     * The view matrix transforms World Space to Camera Space coordinates.
     * \param [in] view Specifies the view matrix.
     */
    virtual void setViewMatrix(vx_matrix view) = 0;

    /**
     * \brief Gets view matrix.
     * The view matrix transforms World Space to Camera Space coordinates.
     * \param [out] view Returns the view matrix.
     */
    virtual void getViewMatrix(vx_matrix view) const = 0;

    /**
     * \brief Sets the projection matrix.
     * The projection matrix transforms Camera Space to Homogeneous Space coordinates.
     * \param [in] projection Specifies the projection matrix.
     */
    virtual void setProjectionMatrix(vx_matrix projection) = 0;
    /**
     * \brief Gets the projection matrix.
     * The projection matrix transforms Camera Space to Homogeneous Space coordinates.
     * \param [out] projection Returns the projection matrix.
     */
    virtual void getProjectionMatrix(vx_matrix projection) const = 0;

    /**
     * \brief Sets the field of view.
     * \param [in] fov Specifies the field of view in degrees.
     */
    virtual void setDefaultFOV(float fov) = 0;

    /**
     * \brief Enables the default keyboard event handler.
     * In this case the user can handle the camera by pressing `W`, `S`, `A`, `D`, `-`, `+` buttons.
     * `W`/`S` - pitch, `A`/`D` - yaw, `-`/`=` - zoom.
     */
    virtual void enableDefaultKeyboardEventCallback() = 0;
    /**
     * \brief Disables default keyboard event handler.
     */
    virtual void disableDefaultKeyboardEventCallback() = 0;
    /**
     * \brief Gets the flag indicating if the default keyboard event handler is enabled.
     * \return `true` indicates enabled; `false` indicates disabled.
     */
    virtual bool useDefaultKeyboardEventCallback() = 0;

    /**
     * \brief Gets the width.
     * \return The width.
     */
    virtual vx_uint32 getWidth() const = 0;
    /**
     * \brief Gets the height.
     * \return The height.
     */
    virtual vx_uint32 getHeight() const = 0;

    /**
     * \brief Puts the image to the render.
     * \param [in] image Specifies the image.
     */
    virtual void putImage(vx_image image) = 0;

    /**
     * \brief Puts a message box on the image.
     * \param [in] text A reference to the text of the message.
     * \param [in] style A reference to the style of the message box.
     */
    virtual void putText(const std::string& text, const ovxio::Render::TextBoxStyle& style) = 0;

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
     * \brief Destructor.
     */
    virtual ~Render3D()
    {}

protected:
    Render3D(TargetType type = Render3D::UNKNOWN_RENDER, const std::string& name = "Undefined"):
        targetType(type),
        renderName(name)
    {}

    const TargetType  targetType;
    const std::string renderName;
};

/**
 * \ingroup group_nvx_render3d
 * \brief Creates Render3D.
 * \param [in] context Specifies the context.
 * \param [in] xPos Specifies the x-coordinate of the top-left corner of the `%Render3D` window.
 * \param [in] yPos Specifies the y-coordinate of the top-left corner of the `%Render3D` window.
 * \param [in] title A reference to the title of the window.
 * \param [in] width Specifies the width of the window.
 * \param [in] height Specifies the height of the window.
 *
 * \see nvxio::createDefaultRender3D
 */
NVXIO_EXPORT std::unique_ptr<Render3D> createDefaultRender3D(vx_context context, int xPos, int yPos,const std::string& title,
                                                             vx_uint32 width, vx_uint32 height);
}

#endif
