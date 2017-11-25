/*
# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NVXCUIO_FRAMESOURCE_HPP
#define NVXCUIO_FRAMESOURCE_HPP

#include <memory>
#include <string>

#include <NVX/nvxcu.h>
#include <NVX/Export.hpp>

/**
 * \file
 * \brief The VisionWorks CUDA `FrameSource` interface and utility functions.
 */

/**
 * \brief Contains VisionWorks CUDA API for image reading and rendering.
 * \ingroup nvx_nvxio_api
 */
namespace nvxio
{

/**
 * \defgroup group_nvxcu_frame_source VisionWorks CUDA FrameSource Interface
 * \ingroup group_nvxio_frame_source
 *
 * This class is designed to be VisionWorks CUDA interface for reading images
 * from different sources.
 *
 */

/**
 * \ingroup group_nvxcu_frame_source
 * \brief %FrameSource interface.
 *
 * Common interface for reading frames from all sources.
 *
 * \note To create the object of this class call the \ref createDefaultFrameSource method.
 *
 * \see nvxio::FrameSource
 */
class NVXIO_EXPORT FrameSource
{
public:
    /**
     * \brief `%FrameSource` parameters.
     */
    struct Parameters
    {
        uint32_t frameWidth; /**< \brief Holds the width of the frame. */
        uint32_t frameHeight; /**< \brief Holds the height of the frame. */
        /**
         * \brief Holds the format of the frame. The default value is `0`.
         * It means that `%FrameSource` selects a color space by its own.
         */
        nvxcu_df_image_e format;
        uint32_t fps; /**< \brief Holds the FPS (for video only). */

        Parameters():
            frameWidth(-1),
            frameHeight(-1),
            format((nvxcu_df_image_e)0),
            fps(-1)
        {}
    };

    /**
     * \brief Defines the type of source.
     */
    enum SourceType
    {
        UNKNOWN_SOURCE, /**< \brief Indicates an unknown source. */
        SINGLE_IMAGE_SOURCE, /**< \brief Indicates a single image. */
        IMAGE_SEQUENCE_SOURCE, /**< \brief Indicates a sequence of images. */
        VIDEO_SOURCE, /**< \brief Indicates a video file. */
        CAMERA_SOURCE /**< \brief Indicates a camera. */
    };

    /**
     * \brief Defines the status of read operations.
     */
    enum FrameStatus
    {
        OK, /**< \brief Indicates the frame has been read successfully. */
        TIMEOUT, /**< \brief Indicates a timeout has been exceeded. */
        CLOSED /**< \brief Indicates the frame source has been closed. */
    };

    /**
     * \brief Opens the FrameSource.
     */
    virtual bool open() = 0;

    /**
     * \brief Fetches frames from the source.
     *
     * The method also performs a color space conversion if `image`'s format is not the same as format of
     * produced frame. `Width` and `height` of the specified `image` must be equal to configuration's ones.
     *
     * \param [out] image                The read image.
     * \param [in]  timeout              Specifies the maximum wait time for the next frame in milliseconds (ms).
     * \return A `%FrameStatus` enumerator.
     */
    virtual FrameSource::FrameStatus fetch(const nvxcu_pitch_linear_image_t & image, uint32_t timeout = 5 /*milliseconds*/) = 0;

    /**
     * \brief Gets the configuration of the `%FrameSource`.
     *
     * \return @ref FrameSource::Parameters describing the current configuration of the `%FrameSource`.
     */
    virtual FrameSource::Parameters getConfiguration() = 0;

    /**
     * \brief Sets the configuration of the `%FrameSource`.
     *
     * The method defines the desired data format, fps, width and height that are assumed to be fetched from the `%FrameSource`.
     * But, it's not guaranteed that the final format is the same as the specified one. Format parameter is treated only as a hint
     * for `%FrameSource` internal implementation to construct more optimal pipeline to avoid future color space conversions.
     *
     * \note FPS, width and height parameters are valid only for V4L2, NvCamera camera sources.
     * For video and image sources they are ignored.
     * \par
     * \note This method can be called only when the `%FrameSource` is not opened. In opposite case, an exception
     * is thrown.
     *
     * \param [in] params A reference to the new configuration of the `%FrameSource`.
     *
     * \return Status of the operation (true - success).
     */
    virtual bool setConfiguration(const FrameSource::Parameters& params) = 0;

    /**
     * \brief Closes the FrameSource.
     */
    virtual void close() = 0;

    /**
     * \brief Destructor.
     */
    virtual ~FrameSource()
    {}

    /**
     * \brief Returns the source type of the `%FrameSource`.
     *
     * \return @ref FrameSource::SourceType.
     */
    FrameSource::SourceType getSourceType() const
    {
        return sourceType;
    }

    /**
     * \brief Returns the source name of the FrameSource.
     *
     * \return Source name.
     */
    std::string getSourceName() const
    {
        return sourceName;
    }

protected:
    FrameSource(FrameSource::SourceType type = FrameSource::UNKNOWN_SOURCE, const std::string & name = "Undefined"):
        sourceType(type),
        sourceName(name)
    {}

    const FrameSource::SourceType  sourceType;
    const std::string sourceName;
};

/**
 * \ingroup group_nvxcu_frame_source
 * \brief FrameSource interface factory that provides appropriate implementation by source URI.
 *
 * \note
 * + It supports the following image formats: PNG, JPEG, JPG, BMP, TIFF.
 * + To read images from a sequence in which, for example, `0000.jpg` corresponds to the first image, `0001.jpg` - the second and so forth,
 *   use the path like this:
 *   `path_to_folder/%04d.jpg`
 * + Use `"device:///v4l2?index=0"` path to capture images from the first camera. It supports only Video for Linux compatible cameras.
 * + Use `"device:///nvcamera"` path to capture frames from NVIDIA GStreamer camera.
 * + Use `"device:///nvmedia?config=dvp-ov10635-yuv422-ab-e2379&number=4"` path to capture frames from Omni Vision cameras via NvMedia.
 * + Image decoding, image sequence decoding, and Video4Linux-compatible camera support require either OpenCV or GStreamer.
 * + Support level of video formats depends on the set of installed GStreamer plugins.
 *
 * \param [in] uri     A reference to the path to image source.
 * \see nvxio::createDefaultFrameSource
 */
NVXIO_EXPORT std::unique_ptr<FrameSource> createDefaultFrameSource(const std::string& uri);

/**
 * \ingroup group_nvxcu_frame_source
 * \brief Loads image from file into OpenVX Image object.
 *
 * The method is a wrapper around FrameSource to simplify single images loading.
 *
 * \param [in] fileName     The path to the image file.
 * \param [in] format       The desired output format.
 * \return `vx_image` object. Calling code is responsible for its releasing.
 *
 * \see nvxio::loadImageFromFile
 */
NVXIO_EXPORT nvxcu_pitch_linear_image_t loadImageFromFile(const std::string& fileName, nvxcu_df_image_e format = NVXCU_DF_IMAGE_RGB);

} // namespace nvxio

#endif // NVXCUIO_FRAMESOURCE_HPP
