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

#ifndef GSTREAMER_UTILS_HPP
#define GSTREAMER_UTILS_HPP

#ifdef USE_GSTREAMER

#include <gst/gst.h>

#define VERSION_NUM(major, minor, micro) (major * 1000000 + minor * 1000 + micro)
#define FULL_GST_VERSION VERSION_NUM(GST_VERSION_MAJOR, GST_VERSION_MINOR, GST_VERSION_MICRO)

#if GST_VERSION_MAJOR == 0
# define COLOR_ELEM "ffmpegcolorspace"
# define ENCODE_ELEM "ffenc_mpeg4"
#elif GST_VERSION_MAJOR == 1
# define COLOR_ELEM "videoconvert"
# define ENCODE_ELEM "avenc_mpeg4"
#else
# error "Your GStreamer version is not supported!"
#endif

namespace nvidiaio {

struct GlibDeleter
{
    void operator ()(void *p) const
    {
        g_free(p);
    }
};

struct GStreamerObjectDeleter
{
    void operator ()(GstBuffer *b) const
    {
        gst_buffer_unref(b);
    }

    void operator ()(GstCaps *c) const
    {
        gst_caps_unref(c);
    }

    void operator ()(GstPad *p) const
    {
        gst_object_unref(p);
    }

    void operator ()(GstMessage *m) const
    {
        gst_message_unref(m);
    }

    void operator ()(GstElementFactory *f) const
    {
        gst_object_unref(f);
    }

#if GST_VERSION_MAJOR > 0
    void operator ()(GstSample *s) const
    {
        gst_sample_unref(s);
    }
#endif
};

} // namespace nvidiaio

#endif // USE_GSTREAMER

#endif // GSTREAMER_UTILS_HPP
