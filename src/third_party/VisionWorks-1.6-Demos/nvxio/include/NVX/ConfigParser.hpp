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

#ifndef NVXIO_CONFIGPARSER_HPP
#define NVXIO_CONFIGPARSER_HPP

#include <string>
#include <memory>

#include "NVX/OptionHandler.hpp"

/**
 * \file
 * \brief The `ConfigParser` interface and utility functions.
 */

namespace nvxio
{
/**
 * \defgroup group_nvxio_config_parser ConfigParser
 * \ingroup nvx_nvxio_api
 *
 * This class is intended for parsing simple text files
 * with a basic structure composed of parameters.
 * Every parameter has a name and a value, delimited by an equal sign (=).
 * The name appears to the left of the equal sign.
 *
 * \par Example File
 * \code
 * name1 = value1
 * name2 = value2
 * name3 = value3
 * \endcode
 */

/**
 * \ingroup group_nvxio_config_parser
 * \brief `%ConfigParser` interface.
 *
 * \see nvx_nvxio_api
 */
class NVXIO_EXPORT ConfigParser
{
public:
    /**
     * \brief Destructor.
     */
    virtual ~ConfigParser(){}

    /**
     * \brief Adds a parameter that should be read.
     *
     * \param [in]  paramName   A reference to the name of the parameter.
     * \param [in]  handler     \ref OptionHandler that will be used to process the parameter.
     */
    virtual void addParameter(const std::string &paramName, OptionHandler::ptr handler)=0;

    /**
     * \brief Parses the configuration file and fills the parameters
     * with the corresponding values from the configuration file.
     *
     * \pre Parameters to be filled have been added using the ConfigParser::addParameter method.
     *
     * \param [in]  pathToConfigFile  A reference to the path to the configuration file.
     *
     * \return Error message or the empty string if the operation has succeeded.
     */
    virtual std::string parse(const std::string &pathToConfigFile)=0;
};

/**
 * \ingroup group_nvxio_config_parser
 * \brief Factory for \ref ConfigParser class.
 *
 * \return The pointer to \c %ConfigParser object.
 *
 * \see nvx_nvxio_api
 */
NVXIO_EXPORT std::unique_ptr<ConfigParser> createConfigParser();

}

#endif // NVXIO_CONFIGPARSER_HPP
