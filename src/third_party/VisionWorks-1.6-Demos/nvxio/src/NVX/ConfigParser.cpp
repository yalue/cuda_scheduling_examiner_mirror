/*
# Copyright (c) 2014-2015, NVIDIA CORPORATION. All rights reserved.
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

#include <NVX/ConfigParser.hpp>

#include <cassert>
#include <iostream>
#include <fstream>
#include <unordered_map>

namespace nvxio
{

namespace {

class ConfigParserImpl: public ConfigParser
{
public:
    void addParameter(const std::string &paramName, OptionHandler::ptr handler);
    std::string parse(const std::string &pathToConfigFile);
private:
    std::unordered_map<std::string, std::unique_ptr<OptionHandler>> parametersList;
};

void ConfigParserImpl::addParameter(const std::string &paramName, OptionHandler::ptr handler)
{
    if (paramName.empty())
    {
        NVXIO_THROW_EXCEPTION("Parameter name should not be empty");
    }

    if (parametersList.find(paramName) != parametersList.end())
    {
        NVXIO_THROW_EXCEPTION("Parameter with the name \"" << paramName << "\" already exists");
    }

    parametersList[paramName] = std::move(handler);
}

std::string trim(const std::string& str)
{
    std::string::size_type firstOfNonSpaces = str.find_first_not_of(" \t");
    std::string::size_type lastOfNonSpaces = str.find_last_not_of(" \t");

    if (firstOfNonSpaces == std::string::npos)
    {
        return std::string();
    }
    else
    {
        return str.substr(firstOfNonSpaces, lastOfNonSpaces - firstOfNonSpaces + 1);
    }
}

bool parseKeyValueString(const std::string &line, std::string &key, std::string &val)
{
    std::string::size_type eqSignPosition = line.find("=");
    if (eqSignPosition != std::string::npos)
    {
        key = trim(line.substr(0, eqSignPosition));
        val = trim(line.substr(eqSignPosition + 1));
        return !key.empty();
    }
    else
    {
        return false;
    }
}


std::string ConfigParserImpl::parse(const std::string &pathToConfigFile)
{
    std::ifstream config(pathToConfigFile);
    std::string line, k, v, msg;
    if (config)
    {
        while (std::getline(config, line))
        {
            line = trim(line);
            if (!line.empty() && !(line[0]=='#'))
            {
                if (parseKeyValueString(line, k, v))
                {
                    auto param = parametersList.find(k);
                    if (param != parametersList.end())
                    {
                        std::string error = param->second->processValue(v);
                        if (!error.empty())
                        {
                            msg += ("Parameter \'"+k+"\' "+error +" (value \'"+v+"\' has been passed)\n");
                        }
                    }
                    else
                    {
                        msg += ("Unknown parameter \'"+ k +"\' has been found\n");
                    }
                }
                else
                {
                    msg += "Encountered an ill-formed line: '" + line + "'; aborting\n";
                    return msg;
                }
            }
        }

        if (config.bad())
        {
            msg += "An I/O error occurred while reading " + pathToConfigFile + "\n";
        }
    }
    else
    {
        msg += ("Path "+ pathToConfigFile +" couldn't be opened!\n");
    }
    return msg;
}

}

std::unique_ptr<ConfigParser> createConfigParser()
{
    return nvxio::makeUP<ConfigParserImpl>();
}

}
