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

#include <cassert>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#include "ArgumentParser.hpp"

namespace nvxio {

namespace {
    std::string makeBooleanOptionHelpString(char shortName, const std::string &longName,
                                            const std::string &description)
    {
        assert(shortName != 0 || !longName.empty());

        std::ostringstream help;

        help << "  ";
        if (shortName == 0)
            help  << "--" << longName;
        else if (longName.empty())
            help << "-" << shortName;
        else
            help << "-" << shortName << ", --" << longName;

        help << "\n    " << description << "\n\n";
        return help.str();
    }

    std::string makeOptionHelpString(char shortName, const std::string &longName,
                                     const std::string &description,
                                     const OptionHandler &handler)
    {
        assert(shortName != 0 || !longName.empty());

        std::ostringstream help;
        std::string placeholder = handler.getPlaceholder();

        help << "  ";
        if (shortName == 0)
            help  << "--" << longName << "=" << placeholder;
        else if (longName.empty())
            help << "-" << shortName << " " << placeholder;
        else
            help << "-" << shortName << " " << placeholder
                 << ", --" << longName << "=" << placeholder;

        help << " (";
        std::string constraint = handler.getConstraintString();
        if (!constraint.empty())
            help << constraint << "; ";
        help << "default: " << handler.getDefaultString() << ")\n";
        help << "    " << description << "\n\n";
        return help.str();
    }
}

ArgumentParser::ArgumentParser() : positionalArgs(nullptr)
{}

void ArgumentParser::addBooleanOption(char shortName, const std::string &longName,
                                      const std::string &description,
                                      bool *result, bool internal)
{
    if (shortName != 0)
    {
        assert(shortBooleanOptions.find(shortName) == shortBooleanOptions.end());
        assert(shortOptions.find(shortName) == shortOptions.end());
        shortBooleanOptions[shortName] = result;
    }

    if (!longName.empty())
    {
        assert(longBooleanOptions.find(longName) == longBooleanOptions.end());
        assert(longOptions.find(longName) == longOptions.end());
        longBooleanOptions[longName] = result;
    }

    (internal ? internalHelpString : externalHelpString) +=
        makeBooleanOptionHelpString(shortName, longName, description);
}

void ArgumentParser::addOption(char shortName, const std::string &longName,
                               const std::string &description,
                               OptionHandler::ptr handler, bool internal)
{
    std::shared_ptr<OptionHandler> sharedHandler(std::move(handler));

    if (shortName != 0)
    {
        assert(shortBooleanOptions.find(shortName) == shortBooleanOptions.end());
        assert(shortOptions.find(shortName) == shortOptions.end());
        shortOptions[shortName] = sharedHandler;
    }

    if (!longName.empty())
    {
        assert(longBooleanOptions.find(longName) == longBooleanOptions.end());
        assert(longOptions.find(longName) == longOptions.end());
        longOptions[longName] = sharedHandler;
    }

    (internal ? internalHelpString : externalHelpString) +=
        makeOptionHelpString(shortName, longName, description, *sharedHandler);
}

void ArgumentParser::allowPositional(std::vector<std::string> *result)
{
    positionalArgs = result;
}

std::string ArgumentParser::getHelpString(bool internal) const
{
    return internal ? internalHelpString : externalHelpString;
}

bool ArgumentParser::parse(int argc, char * argv[]) const
{
    for (const auto &shortBoolOpt: shortBooleanOptions)
        *shortBoolOpt.second = false;

    for (const auto &longBoolOpt: longBooleanOptions)
        *longBoolOpt.second = false;

    if (positionalArgs)
        positionalArgs->clear();

    bool parseSuccessful = true;
    enum { OPTION, OPTION_ARGUMENT, POSITIONAL } expecting = OPTION;
    std::string currentOption;
    OptionHandler *currentOptionHandler = nullptr;

    std::unordered_map<bool *, std::string> seenBooleanOptions;
    std::unordered_map<OptionHandler *, std::string> seenOptions;

    auto handleUnknownOption = [&](const std::string &option)
    {
        std::cerr << argv[0] << ": unknown option: " << option << std::endl;
        parseSuccessful = false;
    };

    auto handleDuplicateOption = [&](const std::string &option, const std::string &previousName)
    {
        std::cerr << argv[0] << ": duplicate option \"" << option << "\"";
        if (option != previousName)
            std::cerr << " (previously specified as \"" << previousName << "\")";
        std::cerr << std::endl;
        parseSuccessful = false;
    };

    auto handleKnownBooleanOption = [&](const std::string &option, bool *valuePointer)
    {
        auto seenIt = seenBooleanOptions.find(valuePointer);
        if (seenIt != seenBooleanOptions.end())
        {
            handleDuplicateOption(option, seenIt->second);
            return;
        }

        seenBooleanOptions.insert({valuePointer, option});
        *valuePointer = true;
    };

    auto handleKnownOption = [&](const std::string &option, const std::string &value, OptionHandler &handler)
    {
        auto seenIt = seenOptions.find(&handler);
        if (seenIt != seenOptions.end())
        {
            handleDuplicateOption(option, seenIt->second);
            return;
        }

        seenOptions.insert({&handler, option});

        std::string error = handler.processValue(value);
        if (!error.empty())
        {
            std::cerr << argv[0] << ": invalid value for option " << option << ": "
                      << error << " (got \"" << value << "\")" << std::endl;
            parseSuccessful = false;
        }
    };

    auto handlePositional = [&](const std::string &arg)
    {
        if (positionalArgs)
        {
            positionalArgs->push_back(arg);
        }
        else
        {
            std::cerr << argv[0] << ": positional arguments are not supported (got \""
                      << arg << "\")" << std::endl;
            parseSuccessful = false;
        }
    };

    for (int argIndex = 1; argIndex < argc; ++argIndex)
    {
        std::string arg = argv[argIndex];

        switch (expecting)
        {
        case OPTION:
            if (arg == "--") // end of option list
            {
                expecting = POSITIONAL;
            }
            else if (arg[0] == '-' && arg[1] == '-') // long option
            {
                std::size_t equalsPos = arg.find('=');

                if (equalsPos == std::string::npos)
                {
                    std::string name = arg.substr(2);
                    auto boolIt = longBooleanOptions.find(name);
                    if (boolIt != longBooleanOptions.end())
                    {
                        handleKnownBooleanOption(arg, boolIt->second);
                        continue;
                    }

                    auto it = longOptions.find(name);
                    if (it != longOptions.end())
                    {
                        currentOption = arg;
                        currentOptionHandler = it->second.get();
                        expecting = OPTION_ARGUMENT;
                        continue;
                    }

                    handleUnknownOption(arg);
                }
                else
                {
                    std::string name = arg.substr(2, equalsPos - 2);
                    std::string nameWithDashes = arg.substr(0, equalsPos);
                    std::string value = arg.substr(equalsPos + 1);

                    auto boolIt = longBooleanOptions.find(name);
                    if (boolIt != longBooleanOptions.end())
                    {
                        std::cerr << argv[0] << ": option " << nameWithDashes
                            << " doesn't take an argument (got \"" << value << "\")" << std::endl;
                        parseSuccessful = false;
                        continue;
                    }

                    auto it = longOptions.find(name);
                    if (it != longOptions.end())
                    {
                        handleKnownOption(nameWithDashes, value, *it->second);
                        continue;
                    }

                    handleUnknownOption(nameWithDashes);
                }
            }
            else if (arg[0] == '-' && arg.size() > 1) // short option block
            {
                for (std::size_t shortOptIndex = 1; shortOptIndex < arg.size(); ++shortOptIndex)
                {
                    char name = arg[shortOptIndex];

                    std::string nameWithDash = {'-', name};

                    auto boolIt = shortBooleanOptions.find(name);
                    if (boolIt != shortBooleanOptions.end())
                    {
                        handleKnownBooleanOption(nameWithDash, boolIt->second);
                        continue;
                    }

                    auto it = shortOptions.find(name);
                    if (it != shortOptions.end())
                    {
                        if (shortOptIndex + 1 == arg.size()) // e.g. -o value
                        {
                            currentOption = nameWithDash;
                            currentOptionHandler = it->second.get();
                            expecting = OPTION_ARGUMENT;
                        }
                        else // e.g. -ovalue
                        {
                            handleKnownOption(nameWithDash, arg.substr(shortOptIndex + 1), *it->second);
                        }
                        break;
                    }

                    handleUnknownOption(nameWithDash);
                }
            }
            else // positional argument
            {
                handlePositional(arg);
            }
            break;
        case OPTION_ARGUMENT:
            handleKnownOption(currentOption, arg, *currentOptionHandler);
            expecting = OPTION;
            break;
        case POSITIONAL:
            handlePositional(arg);
            break;
        }
    }

    if (expecting == OPTION_ARGUMENT)
    {
        std::cerr << argv[0] << ": option " << currentOption << " requires an argument" << std::endl;
        parseSuccessful = false;
    }

    return parseSuccessful;
}

}
