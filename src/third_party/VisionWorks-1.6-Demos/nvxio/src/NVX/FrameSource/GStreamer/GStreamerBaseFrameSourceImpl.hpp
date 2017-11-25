/*
# Copyright (c) 2014-2015, NVIDIA CORPORATION. All rights reserved.
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

#ifndef GSTREAMERBASEFRAMESOURCEIMPL_HPP
#define GSTREAMERBASEFRAMESOURCEIMPL_HPP

#ifdef USE_GSTREAMER

#include "FrameSource/FrameSourceImpl.hpp"

#include "FrameSource/GStreamer/GStreamerCommon.hpp"

// GStreamer pipeline
enum GstAutoplugSelectResult
{
    GST_AUTOPLUG_SELECT_TRY,
    GST_AUTOPLUG_SELECT_EXPOSE,
    GST_AUTOPLUG_SELECT_SKIP
};

namespace nvidiaio
{

class GStreamerBaseFrameSourceImpl :
        public FrameSource
{
public:
    GStreamerBaseFrameSourceImpl(SourceType, const std::string & name);
    virtual bool open();
    virtual FrameStatus fetch(const image_t & image, uint32_t timeout = 5u /*milliseconds*/);
    virtual Parameters getConfiguration();
    virtual bool setConfiguration(const Parameters& params);
    virtual void close();
    virtual ~GStreamerBaseFrameSourceImpl();

    static void newGstreamerPad(GstElement * /*elem*/, GstPad *pad, gpointer data);

protected:

    GStreamerBaseFrameSourceImpl();

    void handleGStreamerMessages();

    virtual bool InitializeGstPipeLine() = 0;
    void FinalizeGstPipeLine();

    GstPipeline*  pipeline;
    GstBus*       bus;

    volatile bool end;

    // Common FrameSource parameters
    int32_t deviceID;
    nvxcu_stream_exec_target_t exec_target;

    Parameters configuration;

    GstElement * sink;

private:

    // temporary CUDA buffer
    void * devMem;
    size_t devMemPitch;

protected:
    static FrameStatus
    extractFrameParams(const Parameters & configuration,
                       GstCaps * bufferCaps, gint & width, gint & height,
                       gint & fps, gint & depth);

#if GST_VERSION_MAJOR == 1
    std::unique_ptr<GstSample, GStreamerObjectDeleter> sampleFirstFrame;
#endif
};

} // namespace nvidiaio

#endif // USE_GSTREAMER

#endif // GSTREAMERBASEFRAMESOURCEIMPL_HPP
