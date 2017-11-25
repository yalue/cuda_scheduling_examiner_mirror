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

#ifndef NVXIO_APPLICATION_HPP
#define NVXIO_APPLICATION_HPP

#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "OptionHandler.hpp"

/**
 * \file
 * \brief The `Application` interface.
 */

/**
 * \brief Contains API for image reading and rendering.
 * \ingroup nvx_nvxio_api
 */
namespace nvxio
{
/**
 * \ingroup nvx_nvxio_api
 * \brief `%Application` interface.
 *
 * This class is intended to simplify the creation of your own application.
 * It performs initialization of global resources, the environment, and parses command line arguments.
 *
 * `%Application` is a singleton. You must call the Application::get static method to get the reference to the object.
 *
 * \see nvx_nvxio_api
 */
class NVXIO_EXPORT Application
{
public:
    /**
     * \brief Defines status codes that your application can return.
     */
    enum ApplicationExitCode
    {
        /** \brief Indicates the operation succeeded. */
        APP_EXIT_CODE_SUCCESS               = 0,
        /** \brief Indicates a generic error code; this code is used when no other code describes the error. */
        APP_EXIT_CODE_ERROR                 = 1,
        /** \brief Indicates an internal or implicit allocation failure. */
        APP_EXIT_CODE_NO_MEMORY             = 2,
        /** \brief Indicates the resource (file, etc.) cannot be acquired. */
        APP_EXIT_CODE_NO_RESOURCE           = 3,
        /** \brief Indicates the framesource exists but cannot be read. */
        APP_EXIT_CODE_NO_FRAMESOURCE        = 4,
        /** \brief Indicates the render cannot be created. */
        APP_EXIT_CODE_NO_RENDER             = 5,
        /** \brief Indicates the supplied graph failed verification. */
        APP_EXIT_CODE_INVALID_GRAPH         = 6,
        /** \brief Indicates the parameter provided does not match the algorithm's possible values or
         * a validation procedure failure. */
        APP_EXIT_CODE_INVALID_VALUE         = 7,
        /** \brief Indicates the parameter provided is too big or too small in dimension, or
         * is not of even size. */
        APP_EXIT_CODE_INVALID_DIMENSIONS    = 8,
        /** \brief Indicates the parameter provided is in an invalid format. */
        APP_EXIT_CODE_INVALID_FORMAT        = 9,
        /** \brief Indicates the object cannot be created. */
        APP_EXIT_CODE_CAN_NOT_CREATE        = 10,
        // add new codes here
    };

    /**
     * \brief Returns the reference to the object.
     * %Application is a singleton. You must call the Application::get static method to get the reference
     * to the object.
     * \return Reference to the object.
     *
     * \par Example Code
     * @snippet nvxio.cpp application_get
     */
    static Application &get();

    /**
     * \brief Destructor.
     */
    virtual ~Application();

    /**
     * \brief Adds Boolean command line option to the application.
     *
     * \param [in]  shortName    Specifies the single-letter name of the option. '\0' means "no short name".
     * \param [in]  longName     A reference to the full name of the option. `nullptr` means "no long name".
     * \param [in]  description  A reference to the description of the option for the console Help message.
     * \param [out] result       A pointer to the variable to set.
     * \a result is `true` if the option is present in the argument list; otherwise, it is `false`.
     *
     * \par Example Code
     * @snippet nvxio.cpp application_add_bool
     */
    virtual void addBooleanOption(char shortName, const std::string &longName,
                                  const std::string &description,
                                  bool *result) = 0;

    /**
     * \brief Adds arbitrary command line option to the application.
     *
     * \param [in] shortName    Specifies the single-letter name of the option. '\0' means "no short name".
     * \param [in] longName     A reference to the full name of the option. `nullptr` means "no long name".
     * \param [in] description  A reference to the description of the option for the console Help message.
     * \param [in] handler      \ref OptionHandler used to process the option.
     *
     * \par Example Code 1
     * @snippet nvxio.cpp application_add_string
     *
     * \par Example Code 2
     * @snippet nvxio.cpp application_add_enum
     */
    virtual void addOption(char shortName, const std::string &longName,
                           const std::string &description,
                           OptionHandler::ptr handler) = 0;

    /**
     * \brief Enables support of positional parameters to be used with the application.
     *
     * \param [in]  placeholder A reference to a short hint that describes the expected values of the parameter.
     * \param [out] result      A pointer to a vector of strings to be
     * filled with the provided parameters.
     *
     * \par Example Code
     * @snippet nvxio.cpp application_allow_positional
     *
     */
    virtual void allowPositionalParameters(const std::string &placeholder,
                                           std::vector<std::string> *result) = 0;

    /**
     * \brief Sets a description of the application for the console Help message.
     *
     * \param [in] description A refernce to the description of the application for the console Help message.
     *
     * \par Example Code
     * @snippet nvxio.cpp application_set_discription
     */
    virtual void setDescription(const std::string &description) = 0;

    /**
     * \brief Initializes the application.
     *
     * \pre The initialization includes parsing command line arguments,
     * so configuration of the command line options must be performed before this call.
     *
     * \param [in] argc A reference to the number of arguments passed into your program from the command line.
     * \param [in] argv A pointer to a pointer to the array of the command line arguments.
     *
     * \par Example Code
     * @snippet nvxio.cpp application_init
     */
    virtual void init(int argc, char **argv) = 0;
    virtual bool initGui() = 0;

    virtual std::string getScenarioName() const = 0;
    virtual int getScenarioLoopCount() const = 0;
    virtual std::string getEventLogName() const = 0;
    virtual bool getEventLogDumpFramesFlag() const = 0;
    virtual bool getVerboseFlag() const = 0;
    virtual bool getFullScreenFlag() const = 0;
    virtual std::string getPreferredRenderName() const = 0;

    /**
     * \brief Finds the file in the sample data directory.
     *
     * This method finds the file in the following directories:
     *
     * + `path_to_your_exe/sources/data/`
     * + `path_to_your_exe/../data/`
     *
     * \note This method throws the `std::runtime_error` exception if the file is not found.
     *
     * \param [in] filename A reference to the name of the required file.
     *
     * \return The absolute path to the file.
     */
    virtual std::string findSampleFilePath(const std::string& filename) const = 0;

    /**
     * \brief Finds the file in the VisionWorks data directory.
     *
     * This method finds the file in the following directories:
     * + `path_to_your_exe/data/`
     *
     * + `VISIONWORKS_DIR/share/visionworks/data/`
     *
     *    Only if `VISIONWORKS_DIR` is defined; `VISIONWORKS_DIR` is a full path to the
     *    installation directory of VisionWorks.
     * + `/usr/share/visionworks/data/`
     *
     *   On Linux and when `VISIONWORKS_DIR` is not defined.
     *
     * \note This method throws the `std::runtime_error` exception if the file is not found.
     *
     * \param [in] filename A reference to the name of the required file.
     *
     * \return The absolute path to the file.
     */
    virtual std::string findLibraryFilePath(const std::string& filename) const = 0;

    virtual int getSourceDefaultTimeout() const = 0;
    virtual void setSourceDefaultTimeout(int timeout) = 0;

    /**
     * \brief Gets a limit for frame rate in frames per second.
     * \return Frame rate limit in frames per second.
    */
    virtual double getFPSLimit() const = 0;
};

}

#endif // NVXIO_APPLICATION_HPP
