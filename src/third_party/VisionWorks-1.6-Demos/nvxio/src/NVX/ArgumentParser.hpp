/*
# Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
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

#ifndef ARGUMENTPARSER_HPP
#define ARGUMENTPARSER_HPP

#include <string>
#include <unordered_map>
#include <vector>

#include <NVX/Application.hpp>

namespace nvxio {

class ArgumentParser
{
public:
    ArgumentParser();

    void addBooleanOption(char shortName, const std::string &longName,
                          const std::string &description,
                          bool *result, bool internal);

    void addOption(char shortName, const std::string &longName,
                   const std::string &description,
                   OptionHandler::ptr handler, bool internal);

    void allowPositional(std::vector<std::string> *result);

    std::string getHelpString(bool internal) const;

    bool parse(int argc, char * argv[]) const;

private:
    std::string externalHelpString;
    std::string internalHelpString;
    std::unordered_map<char, bool*> shortBooleanOptions;
    std::unordered_map<std::string, bool*> longBooleanOptions;
    std::unordered_map<char, std::shared_ptr<OptionHandler>> shortOptions;
    std::unordered_map<std::string, std::shared_ptr<OptionHandler>> longOptions;
    std::vector<std::string> *positionalArgs;
};

}

#endif // ARGUMENTPARSER_HPP
