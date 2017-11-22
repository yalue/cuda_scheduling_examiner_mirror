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

#ifdef USE_GLES

#ifndef EGLAPIACCESSORS_HPP
#define EGLAPIACCESSORS_HPP

#include <EGL/egl.h>
#include <EGL/eglext.h>

#if !defined EGL_KHR_stream || !defined EGL_KHR_stream_fifo || !defined EGL_KHR_stream_consumer_gltexture
# error "EGL_KHR_stream extensions are not supported!"
#endif

namespace nvidiaio
{

class EGLDisplayAccessor
{
public:
    static EGLDisplay getInstance();

private:
    EGLDisplayAccessor();
    ~EGLDisplayAccessor();

    EGLDisplay eglDisplay;
};

#define EXTENSION_LIST_MY(T)                                     \
    T( PFNEGLCREATESTREAMKHRPROC,          eglCreateStreamKHR )  \
    T( PFNEGLDESTROYSTREAMKHRPROC,         eglDestroyStreamKHR ) \
    T( PFNEGLQUERYSTREAMKHRPROC,           eglQueryStreamKHR )   \
    T( PFNEGLSTREAMATTRIBKHRPROC,          eglStreamAttribKHR )

namespace egl_api
{

#define EXTLST_EXTERN(tx, x) extern tx x;

EXTENSION_LIST_MY(EXTLST_EXTERN)

bool setupEGLExtensions();

}

} // namespace nvidiaio

#endif // EGLAPIACCESSORS_HPP

#endif // USE_GLES
