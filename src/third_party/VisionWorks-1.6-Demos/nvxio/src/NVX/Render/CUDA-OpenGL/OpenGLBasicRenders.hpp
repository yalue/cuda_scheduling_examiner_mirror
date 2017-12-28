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

#ifndef OPENGL_BASIC_RENDERS_HPP
#define OPENGL_BASIC_RENDERS_HPP

#include <vector>
#include <sstream>
#include <stdexcept>

#ifdef USE_GUI
# include "OpenGL.hpp"
#endif

#include <ft2build.h>
FT_BEGIN_HEADER
#include FT_FREETYPE_H
FT_END_HEADER

#include <cuda_runtime.h>

#ifndef __ANDROID__
# include "Render/RenderImpl.hpp"
# include "Render/Render3DImpl.hpp"
#else
# include "RenderImpl.hpp"
#endif


#include "Private/LogUtils.hpp"

#ifndef NDEBUG
    void __checkGlError(std::shared_ptr<nvidiaio::GLFunctions> gl_, const char* file, int line);

    #define NVXIO_CHECK_GL_ERROR() __checkGlError(gl_, __FILE__, __LINE__)
#else
    #define NVXIO_CHECK_GL_ERROR() /* nothing */
#endif

namespace nvidiaio
{

class ImageRender
{
public:
    ImageRender();

    bool init(std::shared_ptr<GLFunctions> _gl, uint32_t wndWidth, uint32_t wndHeight);
    void release();

    void render(const image_t & image, uint32_t imageWidth, uint32_t imageHeight);

private:
    void updateTexture(const image_t & image, uint32_t imageWidth, uint32_t imageHeight);
    void renderTexture();

    std::shared_ptr<GLFunctions> gl_;
    GLuint wndWidth_, wndHeight_;

    GLuint tex_[3];
    cudaGraphicsResource_t res_[3];
    GLubyte *host_ptr_[3];
    GLuint vao_;
    GLuint vbo_;

    GLuint pipeline_[3], fragmentProgram_[3];
    GLint index_;
    GLfloat scaleUniformX_, scaleUniformY_;
    bool multiGPUInterop_;
};

class NV12ImageRender
{
public:
    NV12ImageRender();

    bool init(std::shared_ptr<GLFunctions> _gl, uint32_t wndWidth, uint32_t wndHeight);
    void release();

    void render(const image_t & image, uint32_t imageWidth, uint32_t imageHeight);

private:
    void updateTexture(const image_t & image, uint32_t imageWidth, uint32_t imageHeight);
    void renderTexture();

    std::shared_ptr<GLFunctions> gl_;
    GLuint wndWidth_, wndHeight_;

    GLuint tex_[2];
    cudaGraphicsResource_t res_[2];
    GLubyte *host_ptr_[2];
    GLuint vao_;
    GLuint vbo_;

    GLuint program_;
    GLfloat scaleUniformX_, scaleUniformY_;
    bool multiGPUInterop_;
};


class RectangleRender
{
public:
    RectangleRender();

    bool init(std::shared_ptr<GLFunctions> _gl);
    void release();

    void render(const nvxcu_rectangle_t & location, const Render::DetectedObjectStyle& style,
                uint32_t width, uint32_t height, float scale);

private:
    void updateArray(const nvxcu_rectangle_t& location, uint32_t width, uint32_t height, float scale);
    void renderArray(const Render::DetectedObjectStyle& style);

    std::shared_ptr<GLFunctions> gl_;

    GLuint vbo_;
    GLuint vao_;
    GLuint program_;
};

class FeaturesRender
{
public:
    FeaturesRender();

    bool init(std::shared_ptr<GLFunctions> _gl);
    void release();

    void render(const array_t & location, const Render::FeatureStyle& style,
                uint32_t width, uint32_t height, float scaleRatio);
    void render(const array_t & location, const array_t & styles, uint32_t width,
                uint32_t height, float scaleRatio);

private:
    void updateArray(uint32_t start_x, uint32_t end_x, const array_t & location, const array_t & styles);
    void renderArray(uint32_t num_items, uint32_t width, uint32_t height, float scaleRatio,
                     const Render::FeatureStyle& style);

    std::shared_ptr<GLFunctions> gl_;

    uint32_t bufCapacity_;
    GLuint vbo_, vboStyles_;
    GLuint vao_;
    cudaGraphicsResource_t res_, resStyles_;
    GLubyte *host_res_, *host_res_styles_;
    GLuint pipeline_;
    GLuint vertexShaderPoints_, vertexShaderKeyPoints_;
    GLuint vertexShaderPointsPerFeature_, vertexShaderKeyPointsPerFeature_;
    GLuint fragmentShader_, fragmentShaderPerFeature_;

    nvxcu_array_item_type_e currentFeatureType_;
    bool perFeatureStyle_;
    bool multiGPUInterop_;
};

class LinesRender
{
public:
    LinesRender();

    bool init(std::shared_ptr<GLFunctions> _gl);
    void release();

    // renders lines located on GPU
    void render(const array_t & lines, const Render::LineStyle& style,
                uint32_t width, uint32_t height, float scaleRatio);
    // renders lines located on CPU
    void render(const std::vector<nvxcu_point4f_t> & lines, const Render::LineStyle& style,
                uint32_t width, uint32_t height, float scaleRatio);

private:
    // updates internal buffer with lines located on GPU
    void updateArray(uint32_t start_x, uint32_t end_x, const array_t & lines);
    // updates internal buffer with lines located on CPU
    void updateArray(uint32_t start_x, uint32_t end_x, const std::vector<nvxcu_point4f_t> & lines);

    void renderArray(uint32_t num_items, const Render::LineStyle& style,
                     uint32_t width, uint32_t height, float scaleRatio);

    std::shared_ptr<GLFunctions> gl_;
    cudaGraphicsResource_t res_;
    GLubyte *host_ptr_;
    uint32_t bufCapacity_;

    GLuint vbo_[2];
    GLuint vao_[2];
    GLuint program_;
    GLboolean isCPU;
    bool multiGPUInterop_;
};

class ArrowsRender
{
public:
    ArrowsRender();

    bool init(std::shared_ptr<GLFunctions> _gl);
    void release();

    void render(const array_t & old_points, const array_t & new_points, const Render::LineStyle& line_style,
                uint32_t width, uint32_t height, float scaleRatio);

private:
    void updateLinesArray(uint32_t start_x, uint32_t end_x,
                          const array_t & old_points, const array_t & new_points,
                          uint32_t width, uint32_t height, float scaleRatio);

    void renderArray(uint32_t num_items, const Render::LineStyle& style);

    std::shared_ptr<GLFunctions> gl_;
    cudaGraphicsResource_t resOld_, resNew_;
    GLubyte *host_old_ssbo_ptr_, *host_new_ssbo_ptr_;
    uint32_t bufCapacity_;

    GLuint vbo_, ssboOld_, ssboNew_;
    GLuint vao_;
    GLuint program_,
        computeShaderProgramPoints_,
        computeShaderProgramVxKeyPoints_,
        computeShaderProgramNvxKeyPoints_;

    nvxcu_array_item_type_e featureType_;
    bool multiGPUInterop_;
};

class MotionFieldRender
{
public:
    MotionFieldRender();

    bool init(std::shared_ptr<GLFunctions> _gl, uint32_t width, uint32_t height);
    void release();

    void render(const image_t & field, const Render::MotionFieldStyle& style, uint32_t width, uint32_t height, float scaleRatio);

private:
    void updateArray(const image_t & field, uint32_t width, uint32_t height, float scaleRatio);
    void renderArray(const Render::MotionFieldStyle& style);

    std::shared_ptr<GLFunctions> gl_;

    uint32_t capacity_, numPoints_;
    GLuint ssbo_;
    GLuint vao_;
    cudaGraphicsResource_t res_;
    GLfloat *host_ptr_;
    GLuint program_;

    GLuint computeShaderProgram_;
    GLuint ssboTex_;
    bool multiGPUInterop_;
};

class CirclesRender
{
public:
    CirclesRender();

    bool init(std::shared_ptr<GLFunctions> _gl);
    void release();

    void render(const array_t & circles, const Render::CircleStyle& style, uint32_t width, uint32_t height, float scaleRatio);

private:
    void updateArray(const array_t & circles);
    void renderArray(const Render::CircleStyle& style, uint32_t width, uint32_t height, float scaleRatio);

    std::shared_ptr<GLFunctions> gl_;

    std::vector<nvxcu_point4f_t> points_;
    std::vector<uint8_t> tmpArray_;
    uint32_t bufCapacity_;
    GLuint vbo_;
    GLuint vao_;
    GLuint program_;
};

class TextRender
{
public:
    TextRender();
    ~TextRender();

    bool init(std::shared_ptr<GLFunctions> _gl);
    void release();

    void render(const std::string& text, const Render::TextBoxStyle& style,
                uint32_t width, uint32_t height, float scaleRatio);

private:
    std::shared_ptr<GLFunctions> gl_;

    FT_Library ft_;
    FT_Face face_;


    GLuint programBg_;

    GLuint program_;

    GLuint tex_;

    size_t bufCapacity_;
    GLuint vbo_;
    GLuint vboEA_;
    GLuint vao_;

    GLuint bgVbo_;
    GLuint bgVao_;

    struct CharacterInfo
    {
        float ax; // advance.x
        float ay; // advance.y

        float bw; // bitmap.width;
        float bh; // bitmap.rows;

        float bl; // bitmap_left;
        float bt; // bitmap_top;

        float tx; // x offset of glyph in texture coordinates
    } c[128];
    int atlasWidth_, atlasHeight_;

    std::vector<nvxcu_point4f_t> points_;
    std::vector<GLushort> elements_;
};

#ifndef __ANDROID__

class PointCloudRender
{
public:
    PointCloudRender();

    bool init(std::shared_ptr<GLFunctions> _gl);
    void release();

    void render(const array_t & points, const matrix4x4f_t & MVP, const nvidiaio::Render3D::PointCloudStyle& style);

private:
    void updateArray(const matrix4x4f_t & MVP);
    void renderArray(const array_t & points, const nvidiaio::Render3D::PointCloudStyle& style);

    std::shared_ptr<GLFunctions> gl_;

    GLuint pointCloudProgram_;
    GLuint hPointCloudVBO_;
    GLuint hPointCloudVAO_;

    uint32_t bufCapacity_;

    float * dataMVP_;
};

class FencePlaneRender
{
public:
    FencePlaneRender();

    bool init(std::shared_ptr<GLFunctions> _gl);
    void release();

    void render(const array_t & planes, const matrix4x4f_t & MVP, const nvidiaio::Render3D::PlaneStyle & style);

private:
    void updateArray(const array_t & planes, const matrix4x4f_t & MVP);
    void renderArray(const nvidiaio::Render3D::PlaneStyle& style);

    std::shared_ptr<GLFunctions> gl_;

    GLuint fencePlaneProgram_;
    GLuint hFencePlaneVBO_;
    GLuint hFencePlaneEA_;
    GLuint hFencePlaneVAO_;

    uint32_t bufCapacity_;

    float * dataMVP_;

    std::vector<GLfloat> planes_vertices_;
    std::vector<GLushort> planes_elements_;
};

#endif // __ANDROID__

} // namespace nvidiaio

#endif // OPENGL_BASIC_RENDERS_HPP
