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

#ifdef USE_GSTREAMER

#include "FrameSource/GStreamer/GStreamerCommon.hpp"

namespace nvidiaio
{

bool updateConfiguration(GstElement * sink_element, GstElement * color_element,
                         FrameSource::Parameters & configuration)
{
    GList * sink_pads = GST_ELEMENT_PADS(sink_element);
    GList * color_pads = GST_ELEMENT_PADS(color_element);
    GstPad * sink_pad = (GstPad *)g_list_nth_data(sink_pads, 0);
    GstPad * color_pad = (GstPad *)g_list_nth_data(color_pads, 0);

#if GST_VERSION_MAJOR == 0
    std::unique_ptr<GstCaps, GStreamerObjectDeleter> sink_caps(gst_pad_get_caps(sink_pad));
    std::unique_ptr<GstCaps, GStreamerObjectDeleter> color_caps(gst_pad_get_caps(color_pad));
#else
    std::unique_ptr<GstCaps, GStreamerObjectDeleter> sink_caps(gst_pad_get_current_caps(sink_pad));
    std::unique_ptr<GstCaps, GStreamerObjectDeleter> color_caps(gst_pad_get_current_caps(color_pad));
#endif

    if (!color_caps || !sink_caps)
    {
        NVXIO_PRINT("Width, height, fps can not be queried");
        return false;
    }

    const GstStructure * sink_structure = gst_caps_get_structure(sink_caps.get(), 0);
    const GstStructure * color_structure = gst_caps_get_structure(color_caps.get(), 0);

    gint width, height;
    if (!gst_structure_get_int(color_structure, "width", &width))
        NVXIO_PRINT("Cannot query video width");

    if (!gst_structure_get_int(color_structure, "height", &height))
        NVXIO_PRINT("Cannot query video height");

    if (configuration.frameWidth == (vx_uint32)-1)
        configuration.frameWidth = width;
    if (configuration.frameHeight == (vx_uint32)-1)
        configuration.frameHeight = height;

    NVXIO_ASSERT(configuration.frameWidth == static_cast<vx_uint32>(width));
    NVXIO_ASSERT(configuration.frameHeight == static_cast<vx_uint32>(height));

    gint num = 0, denom = 1;
    if (!gst_structure_get_fraction(color_structure, "framerate", &num, &denom))
        NVXIO_PRINT("Cannot query video fps");

    configuration.fps = static_cast<float>(num) / denom;

    // extract actual format
    nvxcu_df_image_e actual_format = NVXCU_DF_IMAGE_NONE;

#if GST_VERSION_MAJOR == 0
    const gchar * name = gst_structure_get_name(sink_structure);
    gint32 bpp = 0;
    guint32 fourcc = 0u;

    if (gst_structure_has_field(sink_structure, "bpp"))
    {
        if (!gst_structure_get_int(sink_structure, "bpp", &bpp))
            NVXIO_PRINT("Cannot query BPP");
    }

    if (gst_structure_has_field(sink_structure, "format"))
    {
        if (!gst_structure_get_fourcc(sink_structure, "format", &fourcc))
            NVXIO_PRINT("Cannot query FOURCC");
    }
#else
    const gchar * str_format = gst_structure_get_string(sink_structure, "format");
#endif
    std::unique_ptr<gchar[], GlibDeleter> element_name(gst_element_get_name(sink_element));

    // nvvidsink is able to produce only RGBX frames
    // while its caps reports about YUV format
    if (element_name && strstr(element_name.get(), "nvvideosink") != nullptr)
        actual_format = NVXCU_DF_IMAGE_RGBX;
    else
    {
#if GST_VERSION_MAJOR == 0
        if (name)
        {
            if (!strcasecmp(name, "video/x-raw-gray"))
            {
                if (bpp == 8)
                    actual_format = NVXCU_DF_IMAGE_U8;
            }
            else if (!strcasecmp(name, "video/x-raw-yuv"))
            {
                if (fourcc == GST_MAKE_FOURCC('N', 'V', '1', '2'))
                    actual_format = NVXCU_DF_IMAGE_NV12;
            }
            else if (!strcasecmp(name, "video/x-raw-rgb"))
            {
                if (bpp == 24)
                    actual_format = NVXCU_DF_IMAGE_RGB;
                else if (bpp == 32)
                    actual_format = NVXCU_DF_IMAGE_RGBX;
            }
        }
#else
        if (str_format)
        {
            if (!strcmp("RGB", str_format))
                actual_format = NVXCU_DF_IMAGE_RGB;
            else if (!strcmp("RGBA", str_format))
                actual_format = NVXCU_DF_IMAGE_RGBX;
            else if (!strcmp("GRAY8", str_format))
                actual_format = NVXCU_DF_IMAGE_U8;
            else if (!strcmp("NV12", str_format))
                actual_format = NVXCU_DF_IMAGE_NV12;
#ifdef USE_GSTREAMER_NVMEDIA
            else if (!strcmp("YV12", str_format))
                actual_format = NVXCU_DF_IMAGE_NV12;
#endif
        }
#endif
    }

    if (actual_format == NVXCU_DF_IMAGE_NONE)
    {
        NVXIO_THROW_EXCEPTION("Unknown format: " <<
#if GST_VERSION_MAJOR == 0
            (name ? name : "(null)")
#else
            (str_format ? str_format : "(null)")
#endif
            );
    }

    // it seems that we should check the equality:
    //     actual_format == configuration.format
    // But, in general case it's impossible to guarantee that
    // actual_format can be achieved (at least for now, since
    // nvvideosink can produce only RGBA frames).
    // So, we threat configuration.format as just a hint for FrameSource.
    configuration.format = actual_format;

    return true;
}

}

#endif // USE_GSTREAMER
