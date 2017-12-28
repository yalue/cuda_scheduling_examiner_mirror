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

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <new>
#include <sstream>

#ifdef __linux__
# include <locale.h>
# include <string.h>
#endif

#ifdef _WIN32
# define NOMINMAX
# include <Windows.h>
#endif

#ifndef __ANDROID__
# include <NVX/Application.hpp>
#endif

namespace nvxio { namespace internal {

#ifndef __ANDROID__
void nvxio_printf(const char * format, ...)
{
    if (nvxio::Application::get().getVerboseFlag())
    {
        std::va_list arg;
        va_start(arg, format);
        std::vprintf(format, arg);
        va_end(arg);
        std::putchar('\n');
    }
}
#endif

#if defined __linux__

class PosixLocale {
public:
    explicit PosixLocale(locale_t base) : locale(duplocale(base))
    {
        if (locale == (locale_t)0) throw std::bad_alloc();
    }

    PosixLocale(const PosixLocale &) = delete;
    PosixLocale & operator = (const PosixLocale &) = delete;

    const char *getErrorString(int errnum)
    {
#ifdef __ANDROID__
        return strerror(errnum);
#else
        return strerror_l(errnum, locale);
#endif
    }

    ~PosixLocale()
    {
        freelocale(locale);
    }

private:
    locale_t locale;
};

std::string errnoToString(int errnum)
{
    std::ostringstream os;
    os << errnum;

    PosixLocale currentLocale(uselocale((locale_t)0));

    if (const char *errorString = currentLocale.getErrorString(errnum))
        os << " - " << errorString;

    return os.str();
}
#endif

#if defined _WIN32
std::string winErrorToString(unsigned long error)
{
    std::ostringstream os;
    os << error;

    char *errorString;

    DWORD length = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM
            | FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_MAX_WIDTH_MASK,
        nullptr, error, 0, (LPSTR)&errorString, 0, nullptr);

    if (length == 0)
    {
        os << " - unknown error";
        return os.str();
    }

    try
    {
        os << " - " << errorString;
        LocalFree(errorString);
    }
    catch (...)
    {
        LocalFree(errorString);
        throw;
    }

    return os.str();
}
#endif

} }
