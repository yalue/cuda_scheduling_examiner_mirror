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

# OS info
OSLOWER := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")

OS_ARCH := $(shell uname -m | sed -e "s/i386/i686/")

# Take command line flags that override any of these settings
ifeq ($(i386),1)
	OS_ARCH := i686
endif

ifeq ($(x86_64),1)
	OS_ARCH := x86_64
endif

ifeq ($(ARMv7),1)
	OS_ARCH := armv7l
endif

ifeq ($(ARMv8),1)
	OS_ARCH := aarch64
endif

CXXFLAGS += -std=c++0x

ifneq ($(VIBRANTE_TOOLCHAIN_SYSROOT),)
	CCFLAGS += --sysroot="$(VIBRANTE_TOOLCHAIN_SYSROOT)"
endif

# Configuration-specific build flags
ifeq ($(dbg),1)
	CCFLAGS += -g -fPIC
	TARGET := debug
else
	CCFLAGS += -O3 -DNDEBUG -fPIC
	TARGET := release
endif

# check visionworks availability
VISION_WORKS_EXISTS := $(shell pkg-config --exists visionworks && echo "1" || echo "0")
ifeq ($(VISION_WORKS_EXISTS), 0)
$(error You must put directory containing visionworks.pc to the PKG_CONFIG_PATH environment variable)
endif

VISION_WORKS_CFLAGS := $(shell pkg-config --cflags visionworks)
CUDA_CFLAGS := $(shell pkg-config --cflags cudart-8.0)

EIGEN_CFLAGS := -I../3rdparty/eigen
NVXIO_CFLAGS := -DCUDA_API_PER_THREAD_DEFAULT_STREAM -DUSE_GUI=1 -DUSE_GLFW=1  -DUSE_GSTREAMER=1
NVXIO_CFLAGS += -I./include  -I../3rdparty/opengl $(shell pkg-config --cflags freetype2) -I../3rdparty/glfw3/include  -I../3rdparty/opengl $(shell pkg-config --cflags gstreamer-base-1.0 gstreamer-pbutils-1.0 gstreamer-app-1.0)

INCLUDES :=
INCLUDES += -Iinclude -Isrc/ -Isrc/NVX/
INCLUDES += $(NVXIO_CFLAGS)
INCLUDES += $(CUDA_CFLAGS)
INCLUDES += $(VISION_WORKS_CFLAGS)
INCLUDES += $(EIGEN_CFLAGS)

VPATH=src/NVX/Private:src/NVX/FrameSource/OpenCV:src/NVX/FrameSource/NvMedia/OV10640:src/NVX/FrameSource/NvMedia/OV10635:src/NVX/FrameSource/NvMedia:src/NVX/FrameSource/GStreamer:src/NVX/FrameSource/Wrappers:src/NVX/FrameSource:src/NVX/Render/OpenCV:src/NVX/Render/GStreamer:src/NVX/Render/CUDA-OpenGL:src/NVX/Render/Wrappers:src/NVX/Render:src/NVX:src/OVX/Private:src/OVX/FrameSource/Wrappers:src/OVX/FrameSource:src/OVX/Render/Wrappers:src/OVX/Render:src/OVX:src:src/NVX/Private:src/NVX/FrameSource/OpenCV:src/NVX/FrameSource/NvMedia/OV10640:src/NVX/FrameSource/NvMedia/OV10635:src/NVX/FrameSource/NvMedia:src/NVX/FrameSource/GStreamer:src/NVX/FrameSource/Wrappers:src/NVX/FrameSource:src/NVX/Render/OpenCV:src/NVX/Render/GStreamer:src/NVX/Render/CUDA-OpenGL:src/NVX/Render/Wrappers:src/NVX/Render:src/NVX:

CPP_FILES :=
CPP_FILES += $(wildcard src/NVX/Private/*.cpp)
CPP_FILES += $(wildcard src/NVX/FrameSource/OpenCV/*.cpp)
CPP_FILES += $(wildcard src/NVX/FrameSource/NvMedia/OV10640/*.cpp)
CPP_FILES += $(wildcard src/NVX/FrameSource/NvMedia/OV10635/*.cpp)
CPP_FILES += $(wildcard src/NVX/FrameSource/NvMedia/*.cpp)
CPP_FILES += $(wildcard src/NVX/FrameSource/GStreamer/*.cpp)
CPP_FILES += $(wildcard src/NVX/FrameSource/Wrappers/*.cpp)
CPP_FILES += $(wildcard src/NVX/FrameSource/*.cpp)
CPP_FILES += $(wildcard src/NVX/Render/OpenCV/*.cpp)
CPP_FILES += $(wildcard src/NVX/Render/GStreamer/*.cpp)
CPP_FILES += $(wildcard src/NVX/Render/CUDA-OpenGL/*.cpp)
CPP_FILES += $(wildcard src/NVX/Render/Wrappers/*.cpp)
CPP_FILES += $(wildcard src/NVX/Render/*.cpp)
CPP_FILES += $(wildcard src/NVX/*.cpp)
CPP_FILES += $(wildcard src/OVX/Private/*.cpp)
CPP_FILES += $(wildcard src/OVX/FrameSource/Wrappers/*.cpp)
CPP_FILES += $(wildcard src/OVX/FrameSource/*.cpp)
CPP_FILES += $(wildcard src/OVX/Render/Wrappers/*.cpp)
CPP_FILES += $(wildcard src/OVX/Render/*.cpp)
CPP_FILES += $(wildcard src/OVX/*.cpp)
CPP_FILES += $(wildcard src/*.cpp)

C_FILES :=
C_FILES += $(wildcard src/NVX/Private/*.c)
C_FILES += $(wildcard src/NVX/FrameSource/OpenCV/*.c)
C_FILES += $(wildcard src/NVX/FrameSource/NvMedia/OV10640/*.c)
C_FILES += $(wildcard src/NVX/FrameSource/NvMedia/OV10635/*.c)
C_FILES += $(wildcard src/NVX/FrameSource/NvMedia/*.c)
C_FILES += $(wildcard src/NVX/FrameSource/GStreamer/*.c)
C_FILES += $(wildcard src/NVX/FrameSource/Wrappers/*.c)
C_FILES += $(wildcard src/NVX/FrameSource/*.c)
C_FILES += $(wildcard src/NVX/Render/OpenCV/*.c)
C_FILES += $(wildcard src/NVX/Render/GStreamer/*.c)
C_FILES += $(wildcard src/NVX/Render/CUDA-OpenGL/*.c)
C_FILES += $(wildcard src/NVX/Render/Wrappers/*.c)
C_FILES += $(wildcard src/NVX/Render/*.c)
C_FILES += $(wildcard src/NVX/*.c)
C_FILES += $(wildcard src/OVX/Private/*.c)
C_FILES += $(wildcard src/OVX/FrameSource/Wrappers/*.c)
C_FILES += $(wildcard src/OVX/FrameSource/*.c)
C_FILES += $(wildcard src/OVX/Render/Wrappers/*.c)
C_FILES += $(wildcard src/OVX/Render/*.c)
C_FILES += $(wildcard src/OVX/*.c)
C_FILES += $(wildcard src/*.c)


NVX_CPP_FILES :=
NVX_CPP_FILES += $(wildcard src/NVX/Private/*.cpp)
NVX_CPP_FILES += $(wildcard src/NVX/FrameSource/OpenCV/*.cpp)
NVX_CPP_FILES += $(wildcard src/NVX/FrameSource/NvMedia/OV10640/*.cpp)
NVX_CPP_FILES += $(wildcard src/NVX/FrameSource/NvMedia/OV10635/*.cpp)
NVX_CPP_FILES += $(wildcard src/NVX/FrameSource/NvMedia/*.cpp)
NVX_CPP_FILES += $(wildcard src/NVX/FrameSource/GStreamer/*.cpp)
NVX_CPP_FILES += $(wildcard src/NVX/FrameSource/Wrappers/*.cpp)
NVX_CPP_FILES += $(wildcard src/NVX/FrameSource/*.cpp)
NVX_CPP_FILES += $(wildcard src/NVX/Render/OpenCV/*.cpp)
NVX_CPP_FILES += $(wildcard src/NVX/Render/GStreamer/*.cpp)
NVX_CPP_FILES += $(wildcard src/NVX/Render/CUDA-OpenGL/*.cpp)
NVX_CPP_FILES += $(wildcard src/NVX/Render/Wrappers/*.cpp)
NVX_CPP_FILES += $(wildcard src/NVX/Render/*.cpp)
NVX_CPP_FILES += $(wildcard src/NVX/*.cpp)

NVX_C_FILES :=
NVX_C_FILES += $(wildcard src/NVX/Private/*.c)
NVX_C_FILES += $(wildcard src/NVX/FrameSource/OpenCV/*.c)
NVX_C_FILES += $(wildcard src/NVX/FrameSource/NvMedia/OV10640/*.c)
NVX_C_FILES += $(wildcard src/NVX/FrameSource/NvMedia/OV10635/*.c)
NVX_C_FILES += $(wildcard src/NVX/FrameSource/NvMedia/*.c)
NVX_C_FILES += $(wildcard src/NVX/FrameSource/GStreamer/*.c)
NVX_C_FILES += $(wildcard src/NVX/FrameSource/Wrappers/*.c)
NVX_C_FILES += $(wildcard src/NVX/FrameSource/*.c)
NVX_C_FILES += $(wildcard src/NVX/Render/OpenCV/*.c)
NVX_C_FILES += $(wildcard src/NVX/Render/GStreamer/*.c)
NVX_C_FILES += $(wildcard src/NVX/Render/CUDA-OpenGL/*.c)
NVX_C_FILES += $(wildcard src/NVX/Render/Wrappers/*.c)
NVX_C_FILES += $(wildcard src/NVX/Render/*.c)
NVX_C_FILES += $(wildcard src/NVX/*.c)


OBJ_DIR := obj/$(TARGET)
OBJ_CPP_FILES := $(addprefix $(OBJ_DIR)/,$(notdir $(CPP_FILES:.cpp=.o)))
OBJ_C_FILES := $(addprefix $(OBJ_DIR)/,$(notdir $(C_FILES:.c=.o)))

OBJ_NVX_CPP_FILES := $(addprefix $(OBJ_DIR)/,$(notdir $(NVX_CPP_FILES:.cpp=.o)))
OBJ_NVX_C_FILES := $(addprefix $(OBJ_DIR)/,$(notdir $(NVX_C_FILES:.c=.o)))

OUTPUT_DIR := ../libs/$(OS_ARCH)/$(OSLOWER)/$(TARGET)$(if $(abi),/$(abi))

################################################################################

# Target rules

all: $(OUTPUT_DIR)/libovx.a  $(OUTPUT_DIR)/libnvx.a

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.o: %.c | $(OBJ_DIR)
	$(CC) $(INCLUDES) $(CCFLAGS) -std=c99 -o $@ -c $<

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(CXX) $(INCLUDES) $(CCFLAGS) $(CXXFLAGS) -o $@ -c $<

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

$(OUTPUT_DIR)/libovx.a: $(OBJ_CPP_FILES) $(OBJ_C_FILES) | $(OUTPUT_DIR)
	$(AR) rsc -o $@ $^

$(OUTPUT_DIR)/libnvx.a: $(OBJ_NVX_CPP_FILES) $(OBJ_NVX_C_FILES) | $(OUTPUT_DIR)
	$(AR) rsc -o $@ $^

clean:
	rm -f $(OBJ_CPP_FILES) $(OBJ_C_FILES)
	rm -f $(OBJ_NVX_CPP_FILES) $(OBJ_NVX_C_FILES)
	rm -fr $(OUTPUT_DIR) $(OBJ_DIR)
