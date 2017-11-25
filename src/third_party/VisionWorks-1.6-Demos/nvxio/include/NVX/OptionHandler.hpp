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

#ifndef NVXIO_OPTIONHANDLER_HPP
#define NVXIO_OPTIONHANDLER_HPP

#include <NVX/Export.hpp>

#include "Range.hpp"

/**
 * \file
 * \brief The `OptionHandler` interface.
 */

namespace nvxio
{
/**
 * \ingroup nvx_nvxio_api
 * \brief `%OptionHandler` interface.
 *
 * This class is an interface supplying a mechanism for processing an option and
 * human-readable information about that option.
 *
 * \see nvx_nvxio_api
 */
class NVXIO_EXPORT OptionHandler
{
private:
    template <typename T>
    struct PairList
    {
        typedef std::initializer_list<std::pair<std::string, T>> type;
    };

public:
    typedef std::unique_ptr<OptionHandler> ptr;

    virtual ~OptionHandler();

    /**
     * \brief Gets a short hint that describes the expected values of the option (i.e., a placeholder for the option).
     * \return Placeholder for the option.
     */
    virtual std::string getPlaceholder() const = 0;

    /**
     * \brief Gets information about valid values of the option.
     * \return Information about valid values of the option.
     */
    virtual std::string getConstraintString() const;

    /**
     * \brief Gets the default value of the option.
     * \return Default value of the option.
     */
    virtual std::string getDefaultString() const = 0;

    /**
     * \brief Processes the value of the option.
     * \param [in] valueStr A reference to the value of the option.
     * \return Status of the operation. If the operation fails returns an error message and an empty string.
     */
    virtual std::string processValue(const std::string &valueStr) const = 0;

    /**
     * \brief Creates an option handler that accepts any argument value and
     * copies it verbatim to the provided variable.
     * \param [in,out] result A pointer to the provided variable that corresponds to the option.
     * \return A pointer to the OptionHandler.
     *
     * \par Example Code
     * @snippet nvxio.cpp application_add_string
     */
    static ptr string(std::string *result);

    /**
     * \brief Creates an option handler that accepts argument values that look like decimal integers,
     * converts them to `int`, and then stores them in the provided variable.
     * \param [in,out] result A pointer to the provided variable that corresponds to the option.
     * \param [in] validRange A reference to a range of valid values of the option.
     * \return A pointer to the OptionHandler.
     *
     * \par Example Code
     * @snippet nvxio.cpp application_add_int
     */
    static ptr integer(int *result, const Range<int> &validRange = ranges::all<int>());

    /**
     * \brief Creates an option handler that accepts argument values that look like unsigned decimal integers,
     * converts them to `unsigned int`, and then stores them in the provided variable.
     * \param [in,out] result A pointer to the provided variable that corresponds to the option.
     * \param [in] validRange A reference to a range of valid values of the option.
     * \return A pointer to the OptionHandler.
     */
    static ptr unsignedInteger(unsigned *result, const Range<unsigned> &validRange = ranges::all<unsigned>());

    /**
     * \brief Creates an option handler that accepts argument values that look like real numbers,
     * converts them to `float`, and then stores them in the provided variable.
     * \param [in,out] result A pointer to the provided variable that corresponds to the option.
     * \param [in] validRange A range of valid values of the option.
     * \return A pointer to the OptionHandler.
     */
    static ptr real(float *result, const Range<float> &validRange = ranges::all<float>());

    /**
     * \brief Creates an option handler that accepts argument values that look like real numbers,
     * converts them to `double`, and then stores them in the provided variable.
     * \param [in,out] result A pointer to the provided variable that corresponds to the option.
     * \param [in] validRange A reference to a range of valid values of the option.
     * \return A pointer to the OptionHandler.
     */
    static ptr real(double *result, const Range<double> &validRange = ranges::all<double>());

    /**
     * \brief Creates an option handler that accepts argument values from a certain set and
     * stores them in the provided variable.
     * \param [in,out] result A pointer to the provided variable that corresponds to the option.
     * \param [in] allowedValues The set of allowed values.
     * \return A pointer to the OptionHandler.
     *
     * \par Example Code
     * @snippet nvxio.cpp application_add_oneof_string
     */
    static ptr oneOf(std::string *result, std::initializer_list<std::string> allowedValues);

    /**
     * \brief Creates an option handler that accepts argument values from a certain set and
     * stores them in the provided variable.
     * \param [in,out] result A pointer to the provided variable that corresponds to the option.
     * \param [in] first Specifies the iterator referring to the `first` element in the container.
     * \param [in] last Specifies the iterator referring to the `past-the-end` element in the container.
     * \return A pointer to the OptionHandler.
     */
    template <typename T, typename It>
    static ptr oneOf(T *result, It first, It last);

    /**
     * \brief Creates an option handler that accepts an argument value from a certain set,
     * maps the argument value to a value of type T, and then stores that value in the provided variable.
     * \param [in,out] result A pointer to the provided variable that corresponds to the option.
     * \param [in] allowedValues The set of allowed values.
     * The set consists of combinations of a key value (`std::string`) and a mapped value
     * that has a type T.
     * \return A pointer to the OptionHandler.
     *
     * \par Example Code
     * @snippet nvxio.cpp application_add_enum
     */

    /* The roundabout way of specifying the second parameter's type is due to an apparent bug in VC++ 2013.
       If specified directly as std::initializer_list<std::pair<std::string, T>>, VC++ insists on deducing
       T from the second argument, and fails. Indirecting it through a metafunction like this is enough to
       persuade it to deduce T from the first argument instead. */
    template <typename T>
    static ptr oneOf(T *result, typename PairList<T>::type allowedValues);
};

}

#include "detail/OptionHandler.hpp"

#endif // NVXIO_OPTIONHANDLER_HPP
