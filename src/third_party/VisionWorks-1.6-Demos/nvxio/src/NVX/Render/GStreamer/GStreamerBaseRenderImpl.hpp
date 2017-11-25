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

#ifndef GSTREAMERBASERENDERIMPL_HPP
#define GSTREAMERBASERENDERIMPL_HPP

#if defined USE_GUI && defined USE_GSTREAMER

#include "Render/GlfwUIRenderImpl.hpp"

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#include "Private/GStreamerUtils.hpp"

#include <sstream>

#define GSTREAMER_DEFAULT_FPS 30

namespace nvidiaio
{

class GStreamerBaseRenderImpl :
        public GlfwUIImpl
{
public:
    GStreamerBaseRenderImpl(TargetType type, const std::string & name);
    virtual bool open(const std::string& path, uint32_t width, uint32_t height, nvxcu_df_image_e format);

    virtual bool flush();
    virtual void close();

    virtual ~GStreamerBaseRenderImpl();

protected:

    virtual bool InitializeGStreamerPipeline() = 0;
    void FinalizeGStreamerPipeline();

    GstPipeline * pipeline;
    GstBus * bus;

    GstAppSrc * appsrc;
    gint64 num_frames;
};

} // namespace nvidiaio

#endif // USE_GUI && USE_GSTREAMER
#endif // GSTREAMERBASERENDERIMPL_HPP
