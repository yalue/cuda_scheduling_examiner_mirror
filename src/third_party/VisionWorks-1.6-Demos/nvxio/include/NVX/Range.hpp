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

#ifndef NVXIO_RANGE_HPP
#define NVXIO_RANGE_HPP

#include <limits>

/**
 * \file
 * \brief The `Range` interface and utility functions.
 */

namespace nvxio
{

/**
 * \defgroup group_nvxio_range Range
 * \ingroup nvx_nvxio_api
 *
 * Defines a range of values.
 */

/**
 * \ingroup group_nvxio_range
 * \brief `%Range` class.
 *
 * \see nvx_nvxio_api
 */
template <typename T>
struct Range
{
    T low;  /**< \brief Holds the left bound of the range. */
    T high; /**< \brief Holds the right bound of the range. */
    bool lowInclusive; /**< \brief Holds the flag that determines if the range includes the left bound.*/
    bool highInclusive; /**< \brief Holds the flag that determines if the range includes the right bound.*/

    /**
     * \brief Determines if the range is left-bounded.
     * A range is left-bounded if there is a number that is smaller than all its elements.
     * \return `true` if the range is left-bounded; otherwise, returns `false`.
     */
    bool lowConstrained() const;

    /**
     * \brief Determines if the range is right-bounded.
     * A range is right-bounded if there is a number that is larger than all its elements.
     * \return `true` if the range is right-bounded; otherwise, returns `false`.
     */
    bool highConstrained() const;

    /**
     * \brief Determines if the range includes the particular point.
     * \return `true` if the range includes the particular poin; otherwise, returns `false`.
     */
    bool contains(T x) const;
};

template <typename T>
inline bool Range<T>::lowConstrained() const
{
    return low != std::numeric_limits<T>::lowest() || !lowInclusive;
}

template <typename T>
inline bool Range<T>::highConstrained() const
{
    return high != std::numeric_limits<T>::max() || !highInclusive;
}

template <typename T>
inline bool Range<T>::contains(T x) const
{
    bool lowOk = lowInclusive ? x >= low : x > low;
    bool highOk = highInclusive ? x <= high : x < high;
    return lowOk && highOk;
}

/**
 * \ingroup group_nvxio_range
 * \brief Calculates intersection of the two ranges.
 * \param [in] r1 A reference to the first range.
 * \param [in] r2 A reference to the second range.
 * \return The intersection of the two ranges.
 */
template <typename T>
inline Range<T> operator & (const Range<T> &r1, const Range<T> &r2)
{
    Range<T> result;

    if (r1.low < r2.low)
    {
        result.low = r2.low;
        result.lowInclusive = r2.lowInclusive;
    }
    else if (r1.low > r2.low)
    {
        result.low = r1.low;
        result.lowInclusive = r1.lowInclusive;
    }
    else
    {
        result.low = r1.low;
        result.lowInclusive = r1.lowInclusive && r2.lowInclusive;
    }

    if (r1.high < r2.high)
    {
        result.high = r1.high;
        result.highInclusive = r1.highInclusive;
    }
    else if (r1.high > r2.high)
    {
        result.high = r2.high;
        result.highInclusive = r2.highInclusive;
    }
    else
    {
        result.high = r2.high;
        result.highInclusive = r1.highInclusive && r2.highInclusive;
    }

    return result;
}

/**
 * \brief Contains API for \ref Range construction.
 * \ingroup group_nvxio_range
 */
namespace ranges
{
/**
 * \ingroup group_nvxio_range
 * \brief Creates a range that includes all points.
 * \return The range.
 */
template <typename T>
inline Range<T> all() {
    return { std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max(), true, true };
}

/**
 * \ingroup group_nvxio_range
 * \brief Creates a range that includes the values that are less than the particular number (values < x).
 * \param [in] x Specifies the right bound of the range (not including the bound).
 * \return The range.
 */
template <typename T>
inline Range<T> lessThan(T x) {
    return { std::numeric_limits<T>::lowest(), x, true, false };
}

/**
 * \ingroup group_nvxio_range
 * \brief Creates a range that includes the values that are greater than the particular number (values > x).
 * \param [in] x Specifies the left bound of the range (not including the bound).
 * \return The range.
 */
template <typename T>
inline Range<T> moreThan(T x) {
    return { x, std::numeric_limits<T>::max(), false, true };
}

/**
 * \ingroup group_nvxio_range
 * \brief Creates a range that includes the values that are greater than or equal to the particular number (values >= x).
 * \param [in] x Specifies the left bound of the range (including the bound).
 * \return The range.
 */
template <typename T>
inline Range<T> atLeast(T x) {
    return { x, std::numeric_limits<T>::max(), true, true };
}

/**
 * \ingroup group_nvxio_range
 * \brief Creates a range that includes the values that are less than or equal to the particular number (values <= x).
 * \param [in] x Specifies the right bound of the range (including the bound).
 * \return The range.
 */
template <typename T>
inline Range<T> atMost(T x) {
    return { std::numeric_limits<T>::lowest(), x, true, true };
}

}

}

#endif // NVXIO_RANGE_HPP
