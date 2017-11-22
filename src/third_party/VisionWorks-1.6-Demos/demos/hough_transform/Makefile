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
CXXFLAGS += -DCUDA_API_PER_THREAD_DEFAULT_STREAM -DUSE_GUI=1 -DUSE_GLFW=1 -DUSE_GSTREAMER=1

ifneq ($(VIBRANTE_TOOLCHAIN_SYSROOT),)
	CCFLAGS += --sysroot="$(VIBRANTE_TOOLCHAIN_SYSROOT)"
endif

# Configuration-specific build flags
ifeq ($(dbg),1)
	CCFLAGS += -g
	TARGET := debug
else
	CCFLAGS += -O3 -DNDEBUG
	TARGET := release
endif

# check visionworks availability
VISION_WORKS_EXISTS := $(shell pkg-config --exists visionworks && echo "1" || echo "0")
ifeq ($(VISION_WORKS_EXISTS), 0)
$(error You must put directory containing visionworks.pc to the PKG_CONFIG_PATH environment variable)
endif

EXTERNAL_CFLAGS :=
EXTERNAL_LIBS :=

EXTERNAL_CFLAGS += $(shell pkg-config --cflags cudart-8.0)
EXTERNAL_LIBS += $(shell pkg-config --libs cudart-8.0)
EXTERNAL_CFLAGS += $(shell pkg-config --cflags visionworks)
EXTERNAL_LIBS += $(shell pkg-config --libs visionworks)


EIGEN_CFLAGS := -I../../3rdparty/eigen

NVXIO_CFLAGS := -I../../nvxio/include -I../../nvxio/src/ -I../../nvxio/src/NVX/
OVXIO_LIBS := ../../libs/$(OS_ARCH)/$(OSLOWER)/$(TARGET)$(if $(abi),/$(abi))/libovx.a

INCLUDES := $(EXTERNAL_CFLAGS)
INCLUDES += -I/home/yang/cuda_scheduling_examiner_mirror/src
INCLUDES += $(NVXIO_CFLAGS)
INCLUDES +=  -I../../3rdparty/opengl -I../../3rdparty/glfw3/include  $(shell pkg-config --cflags freetype2)
INCLUDES += $(EIGEN_CFLAGS)

# to ensure correct linkage with NVIDIA GLES if MESA version is installed
LIBRARIES += -L"$(PKG_CONFIG_SYSROOT_DIR)/usr/lib"

ifneq ($(VIBRANTE_TOOLCHAIN_SYSROOT),)
	LIBRARIES += -L"$(VIBRANTE_TOOLCHAIN_SYSROOT)/usr/lib"
endif

LIBRARIES += $(OVXIO_LIBS)
LIBRARIES += $(shell pkg-config --libs freetype2) ../../3rdparty/glfw3/libs/libglfw3.a $(shell pkg-config --libs gl) $(shell pkg-config --libs xrandr xi xxf86vm x11)
LIBRARIES +=  $(shell pkg-config --libs gstreamer-base-1.0 gstreamer-pbutils-1.0 gstreamer-app-1.0)
LIBRARIES += 
LIBRARIES += $(EXTERNAL_LIBS)
LIBRARIES += 

LDFLAGS += -Wl,--allow-shlib-undefined -pthread

ifneq ($(PKG_CONFIG_SYSROOT_DIR),)
	ifeq ($(ARMv7),1)
		LDFLAGS += -Wl,-rpath-link="$(PKG_CONFIG_SYSROOT_DIR)/lib/arm-linux-gnueabihf"
		LDFLAGS += -Wl,-rpath-link="$(PKG_CONFIG_SYSROOT_DIR)/usr/lib"
		LDFLAGS += -Wl,-rpath-link="$(PKG_CONFIG_SYSROOT_DIR)/usr/lib/arm-linux-gnueabihf"
	endif
endif

# show libraries used by linker in debug mode
ifeq ($(dbg),1)
	LDFLAGS += -Wl,--trace
endif

CUDA_LIB_PATH := $(subst -L$(PKG_CONFIG_SYSROOT_DIR),,$(shell pkg-config --libs-only-L cudart-8.0))
LDFLAGS += -Wl,-rpath=$(CUDA_LIB_PATH)

CPP_FILES := $(wildcard *.cpp)
OBJ_DIR := obj/$(TARGET)
OBJ_FILES_CPP := $(addprefix $(OBJ_DIR)/,$(notdir $(CPP_FILES:.cpp=.o)))

OUTPUT_DIR := ../../../../../bin

################################################################################

# Target rules
all: build

build: $(OUTPUT_DIR)/nvx_demo_hough_transform.so

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(CXX) $(INCLUDES) $(CCFLAGS) $(CXXFLAGS) -fPIC -o $@ -c $<

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

$(OUTPUT_DIR)/nvx_demo_hough_transform.so: $(OBJ_FILES_CPP) $(OVXIO_LIBS) | $(OUTPUT_DIR)
	$(CXX) -shared $(LDFLAGS) $(CCFLAGS) $(CXXFLAGS) -o $@ $^ $(LIBRARIES)

run: build
	./$(OUTPUT_DIR)/nvx_demo_hough_transform.so

clean:
	rm -f $(OBJ_FILES_CPP)
	rm -f $(OUTPUT_DIR)/nvx_demo_hough_transform.so

$(OVXIO_LIBS):
	+@$(MAKE) -C ../../nvxio
