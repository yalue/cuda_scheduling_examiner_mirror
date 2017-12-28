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

#include "OpenGL.hpp"

#ifdef USE_GLFW
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#endif

namespace nvidiaio
{

void loadGLFunctions(GLFunctions *f)
{
#ifdef USE_GLES
#define LOAD(name) f->name = gl ## name
#elif USE_GLFW
#define LOAD(name) f->name = reinterpret_cast<decltype(f->name)>(glfwGetProcAddress("gl" #name))
#else
#define LOAD(name) f->name = nullptr
#endif

    LOAD(ActiveTexture);
    LOAD(AttachShader);
    LOAD(BindBuffer);
    LOAD(BindTexture);
    LOAD(BindVertexArray);
    LOAD(BlendFunc);
    LOAD(BufferData);
    LOAD(Clear);
    LOAD(ClearColor);
    LOAD(CompileShader);
    LOAD(CreateProgram);
    LOAD(CreateShader);
    LOAD(DeleteBuffers);
    LOAD(DeleteProgram);
    LOAD(DeleteShader);
    LOAD(DeleteTextures);
    LOAD(DeleteVertexArrays);
    LOAD(DepthFunc);
    LOAD(Disable);
    LOAD(DisableVertexAttribArray);
    LOAD(DrawArrays);
    LOAD(DrawElements);
    LOAD(Enable);
    LOAD(EnableVertexAttribArray);
    LOAD(GenBuffers);
    LOAD(GenTextures);
    LOAD(GenVertexArrays);
    LOAD(GetAttribLocation);
    LOAD(GetError);
    LOAD(GetProgramInfoLog);
    LOAD(GetProgramiv);
    LOAD(GetShaderInfoLog);
    LOAD(GetShaderiv);
    LOAD(IsBuffer);
    LOAD(IsTexture);
    LOAD(IsVertexArray);
    LOAD(LinkProgram);
    LOAD(MapBufferRange);
    LOAD(ShaderSource);
    LOAD(TexParameterf);
    LOAD(TexParameteri);
    LOAD(TexSubImage2D);
    LOAD(TexImage2D);
    LOAD(Uniform1f);
    LOAD(Uniform1i);
    LOAD(UniformMatrix4fv);
    LOAD(UnmapBuffer);
    LOAD(UseProgram);
    LOAD(ValidateProgram);
    LOAD(VertexAttribPointer);
    LOAD(ReadPixels);
    LOAD(PixelStorei);
    LOAD(IsShader);
    LOAD(IsProgram);
    LOAD(GetFloatv);
    LOAD(LineWidth);
    LOAD(Uniform4f);
    LOAD(BufferSubData);
#ifndef USE_GLES
    LOAD(ClearTexImage);
#endif
    LOAD(DrawArraysInstanced);
    LOAD(VertexAttribDivisor);
    LOAD(GetBooleanv);
    LOAD(DeleteFramebuffers);
    LOAD(IsFramebuffer);
    LOAD(BindFramebuffer);
    LOAD(FramebufferTexture2D);
    LOAD(CheckFramebufferStatus);
    LOAD(GenFramebuffers);
    LOAD(GetIntegerv);
    LOAD(Uniform2f);
    LOAD(DispatchCompute);
    LOAD(BindBufferBase);
    LOAD(BindImageTexture);
    LOAD(MemoryBarrier);
    LOAD(Uniform1ui);
    LOAD(TexStorage2D);
    LOAD(GenProgramPipelines);
    LOAD(DeleteProgramPipelines);
    LOAD(BindProgramPipeline);
    LOAD(UseProgramStages);
    LOAD(CreateShaderProgramv);
    LOAD(ProgramUniform1f);
    LOAD(ProgramUniform2f);
    LOAD(ProgramUniform4f);
    LOAD(GetTexLevelParameteriv);
    LOAD(Viewport);
    LOAD(Hint);
}

} // namespace nvidiaio
