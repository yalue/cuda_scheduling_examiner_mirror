/*
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
*/

#include <cstdlib>
#include <cstring>
#include <iostream>

#include <NVX/Application.hpp>

#include "ArgumentParser.hpp"

#include <chrono>
#include <thread>

#if _WIN32
#define NOMINMAX
#include <Windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef USE_GUI
# define GLFW_INCLUDE_NONE
# include <GLFW/glfw3.h>
#endif

#ifdef USE_GSTREAMER
# include <gst/gst.h>
#endif

#include "Private/LogUtils.hpp"

static const double DEFAULT_FPS_LIMIT = 30;

namespace nvxio {

namespace {

#ifdef _WIN32

class TimerResolutionSetter
{
public:
    TimerResolutionSetter() : setSuccessfully(false), period(0)
    {
        TIMECAPS caps;
        if (timeGetDevCaps(&caps, sizeof(caps)) != MMSYSERR_NOERROR) return;
        period = caps.wPeriodMin;
        if (timeBeginPeriod(period) != TIMERR_NOERROR) return;
        setSuccessfully = true;
    }

    ~TimerResolutionSetter()
    {
        if (setSuccessfully) timeEndPeriod(period);
    }

private:
    TimerResolutionSetter(const TimerResolutionSetter &);
    TimerResolutionSetter &operator =(const TimerResolutionSetter &);

    UINT period;
    bool setSuccessfully;
};

#endif

class ApplicationImpl : public Application
{
public:
    ApplicationImpl();
    ~ApplicationImpl();

    void addBooleanOption(char shortName, const std::string &longName,
                          const std::string &description,
                          bool *result);
    void addOption(char shortName, const std::string &longName,
                   const std::string &description,
                   OptionHandler::ptr handler);
    void allowPositionalParameters(const std::string &placeholder,
                                   std::vector<std::string> *result);

    void setDescription(const std::string &description) { this->description = description; }

    void init(int argc, char **argv);
    bool initGui();

    std::string getScenarioName() const { return scenarioFileName; }
    int getScenarioLoopCount() const { return scenarioLoopCount; }
    std::string getEventLogName() const { return eventLogFileName; }
    bool getEventLogDumpFramesFlag() const { return eventLogDumpFrames; }
    bool getVerboseFlag() const { return verboseFlag; }
    bool getFullScreenFlag() const { return fullScreenFlag; }
    std::string getPreferredRenderName() const { return preferredRenderName; }

    std::string findSampleFilePath(const std::string& filename) const;
    std::string findLibraryFilePath(const std::string& filename) const;

    int getSourceDefaultTimeout() const { return sourceDefaultTimeout; }
    void setSourceDefaultTimeout(int timeout) { sourceDefaultTimeout = timeout; }

    double getFPSLimit() const { return fpsLimit; }
private:

#ifdef USE_GUI
    static void glfwErrorCallback(int /*error*/, const char* description);
#endif

    std::string description;
    std::string eventLogFileName;
    std::string scenarioFileName;
    std::string preferredRenderName;
    int scenarioLoopCount;
    int sourceDefaultTimeout;
    ArgumentParser parser;
    bool helpRequested, nvxioHelpRequested, nvxioFeaturesRequested;
    bool eventLogDumpFrames, verboseFlag, fullScreenFlag;
    std::string positionalPlaceholder;
    bool glfwInitialized;

    double fpsLimit;

#ifdef _WIN32
    TimerResolutionSetter trs; // needed for more accurate sleep
#endif
};

ApplicationImpl::ApplicationImpl()
    : eventLogFileName(""), scenarioFileName(""), preferredRenderName("default"),
      scenarioLoopCount(1), sourceDefaultTimeout(60),
      helpRequested(false), nvxioHelpRequested(false), nvxioFeaturesRequested(false),
      eventLogDumpFrames(false), verboseFlag(false), fullScreenFlag(false),
      glfwInitialized(false), fpsLimit(DEFAULT_FPS_LIMIT)
{
    parser.addBooleanOption('h', "help", "Display this message", &helpRequested, false);
    parser.addBooleanOption(0, "nvxio_help", "Display this message", &nvxioHelpRequested, true);
    parser.addBooleanOption(0, "nvxio_features", "Display the NVXIO library features", &nvxioFeaturesRequested, true);
    parser.addOption(0, "nvxio_scenario_name", "Run events from this scenario",
                     OptionHandler::string(&scenarioFileName), true);
    parser.addOption(0, "nvxio_scenario_loops", "The number of times to loop events from the scenario",
                     OptionHandler::integer(&scenarioLoopCount), true);
    parser.addOption(0, "nvxio_eventlog", "File to log events to",
                     OptionHandler::string(&eventLogFileName), true);
    parser.addOption(0, "nvxio_render", "Default Render type",
#ifdef USE_GUI
                     OptionHandler::oneOf(&preferredRenderName, {"default", "window", "video", "image", "stub"})
#else
                     OptionHandler::oneOf(&preferredRenderName, {"default", "video", "image", "stub"})
#endif
                     , true);
    parser.addOption(0, "nvxio_source_default_timeout", "Default timeout for frame sources",
                     OptionHandler::integer(&sourceDefaultTimeout), true);
    parser.addBooleanOption(0, "nvxio_eventlog_dump_frames", "Dump input frames during writing event log",
                            &eventLogDumpFrames, true);
    parser.addOption(0, "nvxio_fps_limit", "Frame rate limit, in frames per second",
                     nvxio::OptionHandler::real(&fpsLimit, nvxio::ranges::moreThan(0.0)), true);
    parser.addBooleanOption(0, "nvxio_verbose", "Prints internal NVXIO debug messages", &verboseFlag, true);
    parser.addBooleanOption(0, "nvxio_fullscreen", "Run samples and demos in full-screen mode", &fullScreenFlag, true);
}

void ApplicationImpl::addBooleanOption(char shortName, const std::string &longName,
                                       const std::string &description,
                                       bool *result)
{
    parser.addBooleanOption(shortName, longName, description, result, false);
}

void ApplicationImpl::addOption(char shortName, const std::string &longName,
                                const std::string &description,
                                OptionHandler::ptr handler)
{
    parser.addOption(shortName, longName, description, std::move(handler), false);
}

void ApplicationImpl::allowPositionalParameters(const std::string &placeholder,
                               std::vector<std::string> *result)
{
    positionalPlaceholder = placeholder;
    parser.allowPositional(result);
}

#ifdef USE_GUI
void ApplicationImpl::glfwErrorCallback(int /*error*/, const char* description)
{
    NVXIO_PRINT("Glfw callback error: %s", description);
}
#endif

void ApplicationImpl::init(int argc, char **argv)
{
#ifdef USE_GSTREAMER
    if (!gst_is_initialized())
	gst_init(nullptr, nullptr);
#endif
}

bool ApplicationImpl::initGui()
{
#ifdef USE_GUI
    if (!glfwInitialized)
    {
        glfwSetErrorCallback(ApplicationImpl::glfwErrorCallback);
        if (!glfwInit())
        {
            NVXIO_PRINT("Error: Failed to initialize GLFW");
            return false;
        }
        glfwInitialized = true;
    }
#endif

    return glfwInitialized;
}

static std::string getExecPath()
{
#if _WIN32
    char buf[MAX_PATH];
    DWORD len = GetModuleFileName(GetModuleHandle(nullptr), buf, sizeof buf);
    if(len == 0 || len >= sizeof buf)
        NVXIO_THROW_EXCEPTION("Can't determine the executable's location");

    std::string path(buf, len);
    std::replace(path.begin(), path.end(), '\\', '/');
    return path;
#else
    char buf[1024];
    ssize_t len = ::readlink("/proc/self/exe", buf, sizeof(buf));

    if (len <= 0)
        NVXIO_THROW_EXCEPTION("Can't determine the executable's location");

    return std::string(buf, len);
#endif
}

static bool pathIsDirectory(const std::string &path)
{
#if _WIN32
    DWORD attr = GetFileAttributes(path.c_str());
    return attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat pathStat;
    int status = stat(path.c_str(), &pathStat);

    return status == 0 && S_ISDIR(pathStat.st_mode);
#endif
}

static std::string getBinDirPath()
{
    std::string execPath = getExecPath();

    const char bin[] = "/bin/";
    size_t binPos = execPath.rfind(bin);

    if (binPos == std::string::npos)
        NVXIO_THROW_EXCEPTION("Can't find the bin directory");

    return execPath.substr(0, binPos + sizeof bin - 1);
}

static std::string findExistingDir(std::initializer_list<std::string> paths)
{
    auto foundIt = std::find_if(paths.begin(), paths.end(), pathIsDirectory);

    return foundIt == paths.end() ? std::string() : *foundIt;
}

static std::string findSampleDataDir()
{
    std::string binPath = getBinDirPath();

    std::string path = findExistingDir({
        binPath + std::string("sources/data/"),
        binPath + std::string("../data/"),
        //std::string("/home/yang/VisionWorks-1.6-Samples-ECRTS/data/"),
    });

    if (path.empty())
        NVXIO_THROW_EXCEPTION("Can't find the sample data directory");

    return path;
}

std::string ApplicationImpl::findSampleFilePath(const std::string& filename) const
{
    static std::string sampleDataDir = findSampleDataDir();

    return sampleDataDir + filename;
}

static std::string findLibraryDataDir()
{
    std::string binPath = getBinDirPath();

    std::string path = findExistingDir({
        binPath + std::string("data/"),
#if defined(VISIONWORKS_DIR)
        VISIONWORKS_DIR "/share/visionworks/data/",
#elif defined(__linux__)
        binPath + std::string("../library-data/"),
#endif
    });

    if (path.empty())
        NVXIO_THROW_EXCEPTION("Can't find the VisionWorks data directory");

    return path;
}

std::string ApplicationImpl::findLibraryFilePath(const std::string& filename) const
{
    static std::string libraryDataDir = findLibraryDataDir();

    return libraryDataDir + filename;
}

}

ApplicationImpl::~ApplicationImpl()
{
#ifdef USE_GUI
    if (glfwInitialized)
    {
        glfwTerminate();
    }
#endif
#ifdef USE_GSTREAMER
    if (gst_is_initialized())
    {
        gst_deinit();
    }
#endif
}

Application &Application::get()
{
    static ApplicationImpl impl;
    return impl;
}

Application::~Application()
{}

}
