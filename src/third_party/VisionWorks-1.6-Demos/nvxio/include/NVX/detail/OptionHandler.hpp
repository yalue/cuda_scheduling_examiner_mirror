/*
# Copyright (c) 2014, 2016 NVIDIA CORPORATION. All rights reserved.
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

#ifndef NVXIO_DETAIL_OPTIONHANDLER_HPP
#define NVXIO_DETAIL_OPTIONHANDLER_HPP

#include <algorithm>
#include <cassert>
#include <map>

#include "NVX/Utility.hpp"

namespace nvxio { namespace detail {

template <typename T>
class OptionHandlerOneOf : public OptionHandler
{
public:
    template <typename It>
    OptionHandlerOneOf(T *result, It first, It last) : result(result), allowedValues(first, last)
    {
        assert(!allowedValues.empty());
        auto defaultIt = std::find_if(allowedValues.begin(), allowedValues.end(),
            [result](const std::pair<std::string, T> &p) { return p.second == *result; });
        assert(defaultIt != allowedValues.end());
        defaultString = std::string("\"") + defaultIt->first + std::string("\"");
    }

    std::string getPlaceholder() const { return "STRING"; }

    std::string getDefaultString() const { return defaultString; }

    std::string getConstraintString() const
    {
        std::string constraint = "must be one of \"";
        auto it = allowedValues.begin();
        constraint += it->first;
        constraint += "\"";

        for (++it; it != allowedValues.end(); ++it)
        {
            constraint += ", \"";
            constraint += it->first;
            constraint += "\"";
        }

        return constraint;
    }

    std::string processValue(const std::string &valueStr) const {
        auto it = allowedValues.find(valueStr);
        if (it == allowedValues.end()) return getConstraintString();
        *result = it->second;
        return "";
    }

private:
    T *result;
    std::map<std::string, T> allowedValues;
    std::string defaultString;
};

}}

template <typename T>
nvxio::OptionHandler::ptr nvxio::OptionHandler::oneOf(T *result, typename PairList<T>::type allowedValues)
{
    return nvxio::makeUP<detail::OptionHandlerOneOf<T>>(result, allowedValues.begin(), allowedValues.end());
}

template <typename T, typename It>
nvxio::OptionHandler::ptr nvxio::OptionHandler::oneOf(T *result, It first, It last)
{
    typedef typename It::value_type value_type;

    std::vector<std::pair<value_type, value_type>> allowedPairs;

    for (It begin = first; begin != last; ++begin)
    {
        const value_type & v = *begin;
        allowedPairs.emplace_back(v, v);
    }

    return nvxio::makeUP<detail::OptionHandlerOneOf<T>>(result, allowedPairs.begin(), allowedPairs.end());
}

#endif // NVXIO_DETAIL_OPTIONHANDLER_HPP
