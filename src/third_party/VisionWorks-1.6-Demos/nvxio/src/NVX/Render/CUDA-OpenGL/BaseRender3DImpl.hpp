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

#ifndef BASE_RENDER3D_IMPL_HPP
#define BASE_RENDER3D_IMPL_HPP

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#ifdef USE_GUI
# include "Render/CUDA-OpenGL/OpenGL.hpp"
#endif

#include "Render/Render3DImpl.hpp"
#include "Render/CUDA-OpenGL/OpenGLRenderImpl.hpp"
#include "Render/CUDA-OpenGL/OpenGLBasicRenders.hpp"

namespace nvidiaio
{

class BaseRender3DImpl :
        public Render3D
{
public:
    BaseRender3DImpl();

    virtual void putPlanes(const array_t & planes, const matrix4x4f_t & model, const PlaneStyle & style);
    virtual void putPointCloud(const array_t & points, const matrix4x4f_t & model, const PointCloudStyle & style);
    virtual void putImage(const image_t & image);
    virtual void putText(const std::string& text, const Render::TextBoxStyle & style);

    virtual bool flush();

    virtual bool open(int32_t xPos, int32_t yPos, uint32_t windowWidth, uint32_t windowHeight, const std::string & windowTitle);
    virtual void close();

    virtual void setViewMatrix(const matrix4x4f_t & view);
    virtual void getViewMatrix(matrix4x4f_t & view) const;

    virtual void setProjectionMatrix(const matrix4x4f_t & projection);
    virtual void getProjectionMatrix(matrix4x4f_t & projection) const;

    virtual void setDefaultFOV(float fov); // in degrees

    virtual void setOnKeyboardEventCallback(OnKeyboardEventCallback callback, void * context);

    virtual void enableDefaultKeyboardEventCallback();
    virtual void disableDefaultKeyboardEventCallback();
    virtual bool useDefaultKeyboardEventCallback();

    virtual void setOnMouseEventCallback(OnMouseEventCallback callback, void * context);

    virtual uint32_t getWidth() const
    {
        return windowWidth_;
    }

    virtual uint32_t getHeight() const
    {
        return windowHeight_;
    }

    virtual ~BaseRender3DImpl();

protected:
    std::shared_ptr<nvidiaio::GLFunctions> gl_;

    matrix4x4f_t model_;
    matrix4x4f_t view_;
    matrix4x4f_t projection_;

    GLFWwindow * window_;
    std::shared_ptr<OpenGLContextHolder> holder_;

    void initMVP();

    void setModelMatrix(const matrix4x4f_t & model);

    void createOpenGLContextHolder();
    bool initWindow(int32_t xpos, int32_t ypos, uint32_t width, uint32_t height, const std::string& wintitle);

    OnKeyboardEventCallback keyboardCallback_;
    void * keyboardCallbackContext_;

    OnMouseEventCallback mouseCallback_;
    void * mouseCallbackContext_;

    static void mouse_button(GLFWwindow* window, int button, int action, int mods);
    static void cursor_pos(GLFWwindow* window, double x, double y);

    static void keyboardCallbackDefault(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void keyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    bool useDefaultCallback_;

    uint32_t windowWidth_;
    uint32_t windowHeight_;

    uint32_t textureWidth_;
    uint32_t textureHeight_;

    float defaultFOV_;
    const float Z_NEAR_;
    const float Z_FAR_;
    float fov_;

    struct OrbitCameraParams
    {
        const float R_min;
        const float R_max;
        float xr; // rotation angle around x-axis
        float yr; // rotation angle around y-axis
        float R;  // radius of camera orbit

        OrbitCameraParams(float R_min_, float R_max_);

        void applyConstraints();
        void setDefault();
    };

    OrbitCameraParams orbitCameraParams_;

    void updateView();
    void clearGlBuffer();

    nvidiaio::ImageRender imageRender_;
    nvidiaio::NV12ImageRender nv12ImageRender_;
    nvidiaio::TextRender textRender_;
    nvidiaio::PointCloudRender pointCloudRender_;
    nvidiaio::FencePlaneRender fencePlaneRender_;

    float scaleRatio_;
};

} // namespace nvidiaio

#endif // BASE_RENDER3D_IMPL_HPP
