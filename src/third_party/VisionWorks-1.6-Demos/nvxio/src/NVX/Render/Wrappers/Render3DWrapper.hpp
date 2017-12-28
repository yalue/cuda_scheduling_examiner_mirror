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

#ifndef RENDER3D_WRAPPER_NVXCU_HPP
#define RENDER3D_WRAPPER_NVXCU_HPP

#include <memory>

#include <NVX/Render3D.hpp>

#include "Render/Render3DImpl.hpp"

namespace nvxio
{

class Render3DWrapper :
        public Render3D
{
public:
    explicit Render3DWrapper(std::unique_ptr<nvidiaio::Render3D> render);

    virtual void setOnKeyboardEventCallback(OnKeyboardEventCallback callback, void * context);
    virtual void setOnMouseEventCallback(OnMouseEventCallback callback, void * context);

    virtual void putPlanes(const nvxcu_plain_array_t & planes, float * model, const PlaneStyle & style);
    virtual void putPointCloud(const nvxcu_plain_array_t & points, float * model, const PointCloudStyle & style);

    virtual bool flush();
    virtual void close();

    virtual void setViewMatrix(float * view);
    virtual void getViewMatrix(float * view) const;

    virtual void setProjectionMatrix(float * projection);
    virtual void getProjectionMatrix(float * projection) const;

    virtual void setDefaultFOV(float fov);

    virtual void enableDefaultKeyboardEventCallback();
    virtual void disableDefaultKeyboardEventCallback();

    virtual bool useDefaultKeyboardEventCallback();

    virtual uint32_t getWidth() const;
    virtual uint32_t getHeight() const;

    virtual void putImage(const nvxcu_pitch_linear_image_t & image);

    virtual void putText(const std::string & text, const nvxio::Render::TextBoxStyle & style);

    virtual ~Render3DWrapper();

private:
    std::unique_ptr<nvidiaio::Render3D> render_;

    nvidiaio::array_t planes_, points_;
};

} // namespace nvxio

#endif // RENDER3D_WRAPPER_NVXCU_HPP
