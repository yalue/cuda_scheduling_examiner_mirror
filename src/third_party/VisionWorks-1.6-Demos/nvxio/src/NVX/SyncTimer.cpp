/*
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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

#include <cmath>
#include <cstdint>

#if defined __linux__
#include <sys/timerfd.h>
#include <unistd.h>
#elif defined _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

#include <NVX/SyncTimer.hpp>
#include <NVX/Utility.hpp>

#include "Private/LogUtils.hpp"

namespace nvxio {

namespace {

class SyncTimerImpl : public SyncTimer
{
public:
    SyncTimerImpl() : zeroPeriod(false)
    {
#if defined __linux__
        fdTimer = timerfd_create(CLOCK_MONOTONIC, TFD_CLOEXEC);
        if (fdTimer < 0) {
            int error = errno;
            NVXIO_THROW_EXCEPTION("Failed to create a synchronization timer ("
                                  << internal::errnoToString(error) << ")");
        }
#elif defined _WIN32
        hTimer = CreateWaitableTimerW(nullptr, FALSE, nullptr);
        if (!hTimer) {
            DWORD error = GetLastError();
            NVXIO_THROW_EXCEPTION("Failed to create a synchronization timer ("
                                  << internal::winErrorToString(error) << ")");
        }
#endif
    }

    SyncTimerImpl(const SyncTimerImpl &) = delete;
    SyncTimerImpl &operator = (const SyncTimerImpl &) = delete;

    ~SyncTimerImpl()
    {
#if defined __linux__
        close(fdTimer);
#elif defined _WIN32
        CloseHandle(hTimer);
#endif
    }

    virtual void arm(double periodSeconds) {
#if defined __linux__
        double s;
        double ns = std::modf(periodSeconds, &s) * 1e9;

        itimerspec spec = {};
        spec.it_interval.tv_sec = time_t(s);
        spec.it_interval.tv_nsec = long(ns);
        spec.it_value = spec.it_interval;
        if (timerfd_settime(fdTimer, 0, &spec, nullptr) < 0) {
            int error = errno;
            NVXIO_THROW_EXCEPTION("Failed to arm a synchronization timer ("
                                  << internal::errnoToString(error) << ")");
        }

        zeroPeriod = spec.it_interval.tv_sec == 0 && spec.it_interval.tv_nsec == 0;
#elif defined _WIN32
        LARGE_INTEGER zero = {};
        LONG period = static_cast<LONG>(periodSeconds * 1000.);
        if (!SetWaitableTimer(hTimer, &zero, period, nullptr, nullptr, FALSE))
        {
            DWORD error = GetLastError();
            NVXIO_THROW_EXCEPTION("Failed to arm a synchronization timer ("
                                  << internal::winErrorToString(error) << ")");
        }

        zeroPeriod = period == 0;
#endif
    }

    virtual void synchronize() {
        // The system timers we use assume zero period means "no period",
        // so we emulate infinitely fast ticking manually.
        if (zeroPeriod) return;

#if defined __linux__
        std::uint64_t overruns;
        int result;

        do result = read(fdTimer, &overruns, sizeof overruns);
        while (result < 0 && errno == EINTR);

        if (result < 0) {
            int error = errno;
            NVXIO_THROW_EXCEPTION("Failed to synchronize with a timer ("
                                  << internal::errnoToString(error) << ")");
        }
#elif defined _WIN32
        if (WaitForSingleObject(hTimer, INFINITE) == WAIT_FAILED)
        {
            DWORD error = GetLastError();
            NVXIO_THROW_EXCEPTION("Failed to synchronize with a timer ("
                                  << internal::winErrorToString(error) << ")");
        }
#else
#error No implementation for this platform.
#endif
    }

private:
#if defined __linux__
    int fdTimer;
#elif defined _WIN32
    HANDLE hTimer;
#endif
    bool zeroPeriod;
};

}

std::unique_ptr<SyncTimer> createSyncTimer()
{
    return nvxio::makeUP<SyncTimerImpl>();
}

}
