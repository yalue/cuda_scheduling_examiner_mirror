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

#ifdef USE_GUI

#include <Eigen/Dense>

#include <cstring>
#include <cmath>

#include <NVX/Application.hpp>
#include <NVX/ProfilerRange.hpp>

#include "Render/CUDA-OpenGL/BaseRender3DImpl.hpp"

// row-major storage order for compatibility with vx_matrix
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Matrix4f_rm;

//============================================================
// Utility stuff
//============================================================

namespace
{

float toRadians(float degrees)
{
    return degrees * 0.017453292519f;
}

void multiplyMatrix(const nvidiaio::matrix4x4f_t & m1,
                    const nvidiaio::matrix4x4f_t & m2,
                    nvidiaio::matrix4x4f_t & res)
{
    std::memset((void *)res.ptr, 0, sizeof(float) * 16);

    for (int32_t i = 0; i < 4; ++i)
        for (int32_t j = 0; j < 4; ++j)
            for (int32_t k = 0; k < 4; ++k)
                res.ptr[4 *i + j] += m1.ptr[4 * i + k] * m2.ptr[4 * k + j];
}

void calcProjectionMatrix(float fovY, float aspect, float zNear, float zFar, nvidiaio::matrix4x4f_t & projection)
{
    std::memset((void *)projection.ptr, 0, sizeof(float) * 16);

    float ctg = 1.f / tan( fovY/2 );

    projection.ptr[0] = ctg / aspect;
    projection.ptr[5] = ctg;
    projection.ptr[10] = - (zFar + zNear) / (zFar - zNear);
    projection.ptr[11] = - 1.f;
    projection.ptr[14] = - 2.f * zFar * zNear / (zFar - zNear);
}

Matrix4f_rm yawPitchRoll(float yaw, float pitch, float roll)
{
    float tmp_ch = std::cos(yaw);
    float tmp_sh = std::sin(yaw);
    float tmp_cp = std::cos(pitch);
    float tmp_sp = std::sin(pitch);
    float tmp_cb = std::cos(roll);
    float tmp_sb = std::sin(roll);

    Matrix4f_rm Result;
    Result(0,0) = tmp_ch * tmp_cb + tmp_sh * tmp_sp * tmp_sb;
    Result(0,1) = tmp_sb * tmp_cp;
    Result(0,2) = -tmp_sh * tmp_cb + tmp_ch * tmp_sp * tmp_sb;
    Result(0,3) = 0.0f;
    Result(1,0) = -tmp_ch * tmp_sb + tmp_sh * tmp_sp * tmp_cb;
    Result(1,1) = tmp_cb * tmp_cp;
    Result(1,2) = tmp_sb * tmp_sh + tmp_ch * tmp_sp * tmp_cb;
    Result(1,3) = 0.0f;
    Result(2,0) = tmp_sh * tmp_cp;
    Result(2,1) = -tmp_sp;
    Result(2,2) = tmp_ch * tmp_cp;
    Result(2,3) = 0.0f;
    Result(3,0) = 0.0f;
    Result(3,1) = 0.0f;
    Result(3,2) = 0.0f;
    Result(3,3) = 1.0f;

    return Result;
}

void lookAt(nvidiaio::matrix4x4f_t & view, const Eigen::Vector3f& eye, const Eigen::Vector3f& center, const Eigen::Vector3f& up)
{
    Eigen::Vector3f f = Eigen::Vector3f(center - eye).normalized();
    Eigen::Vector3f s = f.cross(up).normalized();
    Eigen::Vector3f u = s.cross(f);

    Matrix4f_rm result = Matrix4f_rm::Identity();
    result(0,0) = s(0);
    result(1,0) = s(1);
    result(2,0) = s(2);
    result(0,1) = u(0);
    result(1,1) = u(1);
    result(2,1) = u(2);
    result(0,2) = -f(0);
    result(1,2) = -f(1);
    result(2,2) = -f(2);
    result(3,0) = -s.dot(eye);
    result(3,1) = -u.dot(eye);
    result(3,2) = f.dot(eye);

    std::memcpy(view.ptr, result.data(), sizeof(float) * 16);
}

void updateOrbitCamera(nvidiaio::matrix4x4f_t & view, float xr, float yr, float distance, const Eigen::Vector3f & target)
{
    Matrix4f_rm R = yawPitchRoll(xr, yr, 0.0f);

    Eigen::Vector3f T(0.0f, 0.0f, -distance);

    Eigen::Vector4f T_ = R * Eigen::Vector4f(T(0), T(1), T(2), 0.0f);

    T = Eigen::Vector3f(T_(0), T_(1), T_(2));

    Eigen::Vector3f position = target + T;

    Eigen::Vector4f up_ = R*Eigen::Vector4f(0.0f, -1.0f, 0.0f, 0.0f);

    Eigen::Vector3f up(up_[0], up_[1], up_[2]);

    lookAt(view, position, target, up);
}

void matrixSetEye(nvidiaio::matrix4x4f_t & m)
{
    static const float data[4*4] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    std::memcpy(m.ptr, data, sizeof(data));
}

} // namespace

//============================================================
// Callbacks and events
//============================================================

void nvidiaio::BaseRender3DImpl::enableDefaultKeyboardEventCallback()
{
    if (!useDefaultCallback_)
    {
        useDefaultCallback_ = true;

        fov_ = defaultFOV_;
        initMVP();
    }
}

void nvidiaio::BaseRender3DImpl::disableDefaultKeyboardEventCallback()
{
    if(useDefaultCallback_)
    {
        useDefaultCallback_ = false;

        fov_ = defaultFOV_;
        orbitCameraParams_.setDefault();
        initMVP();
    }
}

bool nvidiaio::BaseRender3DImpl::useDefaultKeyboardEventCallback()
{
    return useDefaultCallback_;
}

void nvidiaio::BaseRender3DImpl::setOnKeyboardEventCallback(nvidiaio::Render3D::OnKeyboardEventCallback callback, void * context)
{
    keyboardCallback_ = callback;
    keyboardCallbackContext_ = context;

    glfwSetKeyCallback(window_, keyboardCallback);
}

void nvidiaio::BaseRender3DImpl::keyboardCallbackDefault(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/)
{
    nvidiaio::BaseRender3DImpl* impl = static_cast<nvidiaio::BaseRender3DImpl*>(glfwGetWindowUserPointer(window));

    if (action == GLFW_PRESS)
    {
        static const float stepAngle = toRadians(4); // in radians
        static const float stepR = 1;

        switch (key)
        {
            case GLFW_KEY_ESCAPE:
            {
                glfwSetWindowShouldClose(window, 1);
                break;
            }
            case GLFW_KEY_MINUS:
            {
                impl->orbitCameraParams_.R += stepR;
                break;
            }
            case GLFW_KEY_EQUAL:
            {
                impl->orbitCameraParams_.R -= stepR;
                break;
            }

            case GLFW_KEY_A:
            {
                impl->orbitCameraParams_.xr -= stepAngle;
                break;
            }
            case GLFW_KEY_D:
            {
                impl->orbitCameraParams_.xr += stepAngle;
                break;
            }
            case GLFW_KEY_W:
            {
                impl->orbitCameraParams_.yr += stepAngle;
                break;
            }
            case GLFW_KEY_S:
            {
                impl->orbitCameraParams_.yr -= stepAngle;
                break;
            }
        }
        impl->updateView();
    }
}

void nvidiaio::BaseRender3DImpl::keyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    nvidiaio::BaseRender3DImpl* impl = static_cast<nvidiaio::BaseRender3DImpl*>(glfwGetWindowUserPointer(window));

    if (action == GLFW_PRESS)
    {
        double x, y;
        glfwGetCursorPos(window, &x, &y);

        if (key == GLFW_KEY_ESCAPE)
            key = 27;

        if(impl->keyboardCallback_)
            (impl->keyboardCallback_)(impl->keyboardCallbackContext_, tolower(key),
                                      static_cast<uint32_t>(x),
                                      static_cast<uint32_t>(y));

        if (impl->useDefaultKeyboardEventCallback())
        {
            keyboardCallbackDefault(window, key, scancode, action, mods);
        }
    }
}

void nvidiaio::BaseRender3DImpl::setOnMouseEventCallback(OnMouseEventCallback callback, void * context)
{
    mouseCallback_ = callback;
    mouseCallbackContext_ = context;

    glfwSetMouseButtonCallback(window_, mouse_button);
    glfwSetCursorPosCallback(window_, cursor_pos);
}

void nvidiaio::BaseRender3DImpl::mouse_button(GLFWwindow* window, int button, int action, int /*mods*/)
{
    BaseRender3DImpl* impl = static_cast<BaseRender3DImpl*>(glfwGetWindowUserPointer(window));

    if (impl->mouseCallback_)
    {
        Render3D::MouseButtonEvent event = nvxio::Render3D::MouseMove;

        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            if (action == GLFW_RELEASE)
                event = nvxio::Render3D::LeftButtonUp;
            else
                event = nvxio::Render3D::LeftButtonDown;
        }
        if (button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            if (action == GLFW_RELEASE)
                event = nvxio::Render3D::RightButtonUp;
            else
                event = nvxio::Render3D::RightButtonDown;
        }
        if (button == GLFW_MOUSE_BUTTON_MIDDLE)
        {
            if (action == GLFW_RELEASE)
                event = nvxio::Render3D::MiddleButtonUp;
            else
                event = nvxio::Render3D::MiddleButtonDown;
        }

        double x = 0.0, y = 0.0;
        glfwGetCursorPos(window, &x, &y);
        (impl->mouseCallback_)(impl->mouseCallbackContext_, event,
                               static_cast<uint32_t>(x),
                               static_cast<uint32_t>(y));
    }
}

void nvidiaio::BaseRender3DImpl::cursor_pos(GLFWwindow* window, double x, double y)
{
    BaseRender3DImpl* impl = static_cast<BaseRender3DImpl*>(glfwGetWindowUserPointer(window));

    if (impl->mouseCallback_)
        (impl->mouseCallback_)(impl->mouseCallbackContext_, nvxio::Render3D::MouseMove,
                               static_cast<uint32_t>(x),
                               static_cast<uint32_t>(y));
}

//============================================================
// Orbit Camera Params
//============================================================

nvidiaio::BaseRender3DImpl::OrbitCameraParams::OrbitCameraParams(float R_min_, float R_max_) :
    R_min(R_min_), R_max(R_max_)
{
    setDefault();
}

void nvidiaio::BaseRender3DImpl::OrbitCameraParams::applyConstraints()
{
    if (R < R_min) R = R_min;
    else if (R > R_max) R = R_max;

    if (yr > 2 * ovxio::PI_F) yr -= 2 * ovxio::PI_F;
    else if (yr < 0) yr += 2 * ovxio::PI_F;

    if (xr > 2 * ovxio::PI_F) xr -= 2 * ovxio::PI_F;
    else if (xr < 0) xr += 2 * ovxio::PI_F;
}

void nvidiaio::BaseRender3DImpl::OrbitCameraParams::setDefault()
{
    R = R_min;
    xr = 0.0f;
    yr = 0.0f;
}

//============================================================
// Setters and getters
//============================================================

void nvidiaio::BaseRender3DImpl::setDefaultFOV(float fov)
{
    defaultFOV_ = std::max(std::min(fov, 180.0f), 0.0f);
}

void nvidiaio::BaseRender3DImpl::setViewMatrix(const matrix4x4f_t & view)
{
    view_ = view;
}

void nvidiaio::BaseRender3DImpl::setProjectionMatrix(const matrix4x4f_t & projection)
{
    projection_ = projection;
}

void nvidiaio::BaseRender3DImpl::getViewMatrix(matrix4x4f_t & view) const
{
    view = view_;
}

void nvidiaio::BaseRender3DImpl::getProjectionMatrix(matrix4x4f_t & projection) const
{
    projection = projection_;
}

void nvidiaio::BaseRender3DImpl::setModelMatrix(const matrix4x4f_t & model)
{
    model_ = model;
}

//============================================================
// Put objects to OpenGL framebuffer
//============================================================

void nvidiaio::BaseRender3DImpl::putPointCloud(const array_t & points, const matrix4x4f_t & model, const PointCloudStyle& style)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render3D::putPointCloud (NVXIO)");

    setModelMatrix(model);

    // MVP = model * view * projection
    matrix4x4f_t MVP;

    multiplyMatrix(model_, view_, MVP);
    multiplyMatrix(matrix4x4f_t(MVP), projection_,  MVP);

    OpenGLContextSafeSetter setter(holder_);

    // invoke
    pointCloudRender_.render(points, MVP, style);
}

void nvidiaio::BaseRender3DImpl::putPlanes(const array_t & planes, const matrix4x4f_t & model, const nvidiaio::Render3D::PlaneStyle& style)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render3D::putPlanes (NVXIO)");

    setModelMatrix(model);

    // MVP = model * view * projection
    matrix4x4f_t MVP;

    multiplyMatrix(model_, view_, MVP);
    multiplyMatrix(matrix4x4f_t(MVP), projection_,  MVP);

    OpenGLContextSafeSetter setter(holder_);

    // Invoke
    fencePlaneRender_.render(planes, MVP, style);
}

void nvidiaio::BaseRender3DImpl::putImage(const image_t & image)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render3D::putImage (NVXIO)");

    // calculate actual ScaleRatio that will be applied to other primitives like lines, circles, etc.

    uint32_t imageWidth = image.width, imageHeight = image.height;

    float widthRatio = static_cast<float>(windowWidth_) / imageWidth;
    float heightRatio = static_cast<float>(windowHeight_) / imageHeight;
    scaleRatio_ = std::min(widthRatio, heightRatio);

    textureWidth_ = static_cast<uint32_t>(scaleRatio_ * imageWidth);
    textureHeight_ = static_cast<uint32_t>(scaleRatio_ * imageHeight);

    OpenGLContextSafeSetter setter(holder_);

    if (image.format == NVXCU_DF_IMAGE_NV12)
        nv12ImageRender_.render(image, textureWidth_, textureHeight_);
    else
        imageRender_.render(image, textureWidth_, textureHeight_);
}

void nvidiaio::BaseRender3DImpl::putText(const std::string& text, const nvidiaio::Render::TextBoxStyle& style)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render3D::putText (NVXIO)");

    OpenGLContextSafeSetter setter(holder_);
    textRender_.render(text, *(const nvidiaio::Render::TextBoxStyle *)&style, textureWidth_, textureHeight_, scaleRatio_);
}

//============================================================
// Initialize and deinitialize
//============================================================

nvidiaio::BaseRender3DImpl::BaseRender3DImpl():
    nvidiaio::Render3D(nvxio::Render3D::BASE_RENDER_3D, "BaseOpenGlRender3D"),
    gl_(nullptr),
    model_(),
    view_(),
    projection_(),
    window_(nullptr),
    holder_(nullptr),
    keyboardCallback_(nullptr),
    keyboardCallbackContext_(nullptr),
    useDefaultCallback_(true),
    windowWidth_(0u),
    windowHeight_(0u),
    textureWidth_(0u),
    textureHeight_(0u),
    defaultFOV_(70),// in degrees
    Z_NEAR_(0.01f),
    Z_FAR_(500.0f),
    fov_(defaultFOV_),
    orbitCameraParams_(1e-6f, 100.f)
{
}

void nvidiaio::BaseRender3DImpl::initMVP()
{
    matrixSetEye(model_);

    const static float viewData[4*4] = {1, 0, 0, 0,
                                        0, -1, 0, 0,
                                        0, 0, -1, 0,
                                        0, 0, 0, 1};

    std::memcpy(view_.ptr, viewData, sizeof(viewData));

    calcProjectionMatrix(toRadians(fov_), (float)windowWidth_ / windowHeight_, Z_NEAR_, Z_FAR_, projection_);
}

void nvidiaio::BaseRender3DImpl::updateView()
{
    orbitCameraParams_.applyConstraints();
    updateOrbitCamera(view_, orbitCameraParams_.xr, orbitCameraParams_.yr, orbitCameraParams_.R, Eigen::Vector3f(0, 0, 0));
}

namespace {

class GLFWContextHolderImpl :
        public nvidiaio::OpenGLContextHolder
{
public:
    explicit GLFWContextHolderImpl(GLFWwindow * currentWindow_) :
        prevWindow(nullptr), currentWindow(currentWindow_)
    {
        if (!currentWindow)
            NVXIO_THROW_EXCEPTION("The render is closed, you must open it before");
    }

    virtual void set()
    {
        // save current context
        prevWindow = glfwGetCurrentContext();
        // attach our OpenGL context
        glfwMakeContextCurrent(currentWindow);
    }

    virtual void unset()
    {
        // attach previous context
        glfwMakeContextCurrent(prevWindow);
    }

private:
    GLFWwindow * prevWindow, * currentWindow;
};

}

void nvidiaio::BaseRender3DImpl::createOpenGLContextHolder()
{
    holder_ = std::make_shared<GLFWContextHolderImpl>(window_);
}

bool nvidiaio::BaseRender3DImpl::open(int32_t xPos, int32_t yPos, uint32_t windowWidth, uint32_t windowHeight, const std::string& windowTitle)
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render3D::open (NVXIO)");

    windowWidth_ = windowWidth;
    windowHeight_ = windowHeight;

    if (!nvxio::Application::get().initGui())
    {
        NVXIO_PRINT("Error: Failed to init GUI");
        return false;
    }

    bool result = initWindow(xPos, yPos, windowWidth_, windowHeight_, windowTitle.c_str());

    fov_ = defaultFOV_;
    initMVP();

    return result;
}

bool nvidiaio::BaseRender3DImpl::initWindow(int32_t xpos, int32_t ypos, uint32_t width, uint32_t height, const std::string& wintitle)
{
    int count = 0;
    GLFWmonitor ** monitors = glfwGetMonitors(&count);
    if (count == 0)
    {
        NVXIO_THROW_EXCEPTION("GLFW: no monitors found");
    }

    int maxPixels = 0;
    const GLFWvidmode* mode = nullptr;

    for (int i = 0; i < count; ++i)
    {
        const GLFWvidmode* currentMode = glfwGetVideoMode(monitors[i]);
        int currentPixels = currentMode->width * currentMode->height;

        if (maxPixels < currentPixels)
        {
            mode = currentMode;
            maxPixels = currentPixels;
        }
    }

    uint32_t cur_width = 0u, cur_height = 0u;
    if ((width <= (uint32_t)mode->width) && (height <= (uint32_t)mode->height))
    {
        cur_width = width;
        cur_height = height;
    }
    else
    {
        float widthRatio = static_cast<float>(mode->width) / width;
        float heightRatio = static_cast<float>(mode->height) / height;
        float scaleRatio = std::min(widthRatio, heightRatio);
        cur_width = static_cast<uint32_t>(scaleRatio * width);
        cur_height = static_cast<uint32_t>(scaleRatio * height);
    }

    glfwWindowHint(GLFW_RESIZABLE, 0);
#ifdef USE_GLES
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif
    window_ = glfwCreateWindow(cur_width, cur_height,
                               wintitle.c_str(),
                               nullptr, nullptr);
    if (!window_)
    {
        NVXIO_PRINT("Error: Failed to create GLFW window");
        return false;
    }

    glfwSetWindowUserPointer(window_, this);
    glfwSetWindowPos(window_, xpos, ypos);
    glfwSetInputMode(window_, GLFW_STICKY_KEYS, GL_TRUE);

    // Create OpenGL context holder

    createOpenGLContextHolder();
    OpenGLContextSafeSetter setter(holder_);

    // Initialize OpenGL renders

    if (!gl_)
    {
        gl_ = std::make_shared<nvidiaio::GLFunctions>();
        nvidiaio::loadGLFunctions(gl_.get());
    }

    if (!imageRender_.init(gl_, width, height))
        return false;

    if (!nv12ImageRender_.init(gl_, width, height))
        return false;

    if (!textRender_.init(gl_))
        return false;

    if (!pointCloudRender_.init(gl_))
        return false;

    if (!fencePlaneRender_.init(gl_))
        return false;

    return true;
}

bool nvidiaio::BaseRender3DImpl::flush()
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render3D::flush (NVXIO)");

    if (!window_)
        NVXIO_THROW_EXCEPTION("The render is closed, you must open it before");

    if (glfwWindowShouldClose(window_))
    {
        close();
        return false;
    }

    // GLFW says that we don't need current OpenGL context, but
    // it's wrong for EGL (OpenGL ES).
    // See EGL 1.4 spec. 3.9.3. Posting Semantics;
    // See EGL 1.5 spec. 3.10.3. Posting Semantics;
    {
        OpenGLContextSafeSetter setter(holder_);
        glfwSwapBuffers(window_);
    }

    glfwPollEvents();

    clearGlBuffer();

    return true;
}

void nvidiaio::BaseRender3DImpl::clearGlBuffer()
{
    OpenGLContextSafeSetter setter(holder_);

    gl_->ClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    NVXIO_CHECK_GL_ERROR();
    gl_->Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    NVXIO_CHECK_GL_ERROR();
}

void nvidiaio::BaseRender3DImpl::close()
{
    nvxio::ProfilerRange range(nvxio::COLOR_ARGB_FUSCHIA, "Render3D::close (NVXIO)");

    if (window_)
    {
        OpenGLContextSafeSetter setter(holder_);

        textRender_.release();
        imageRender_.release();
        nv12ImageRender_.release();
        pointCloudRender_.release();
        fencePlaneRender_.release();

        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
}

nvidiaio::BaseRender3DImpl::~BaseRender3DImpl()
{
    close();
}

#endif
