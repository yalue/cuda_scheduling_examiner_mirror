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

#ifndef NVXIO_THREADSAFEQUEUE_HPP
#define NVXIO_THREADSAFEQUEUE_HPP

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <queue>
#include <mutex>

/**
 * \file
 * \brief The `ThreadSafeQueue` class.
 */

namespace nvxio
{

/**
 * \defgroup group_nvxio_thread_safe_queue Thread-Safe Queue
 * \ingroup nvx_nvxio_api
 *
 * Defines a thread-safe queue.
 */

/**
 * \ingroup group_nvxio_thread_safe_queue
 * \brief A maximum timeout.
 */
const unsigned int TIMEOUT_INFINITE = 0xFFFFFFFF;

/**
 * \ingroup group_nvxio_thread_safe_queue
 * \brief `Thread-Safe Queue` class.
 *
 * \see nvx_nvxio_api
 */
template <typename T>
class ThreadSafeQueue
{
public:

    /**
     * \brief Creation of a thread-safe queue with a specified capacity.
     *
     * \param [in]  maxSize     A capacity of a queue.
     */
    explicit ThreadSafeQueue(std::size_t maxSize) : maxSize(maxSize)
    {}

    /**
     * \brief Attemps to push a new value into the queue.
     *
     * \param [in] item         A new value to push into the queue.
     * \param [in] timeout      A maximum timeout in ms that the method should be waiting while the queue is full.
     *
     * \return Status of the operation. Returns `true` in case of a new element has been pushed into the queue.
     * Otherwise the method returns `false`.
     */
    bool push(const T& item, unsigned int timeout = 1 /*milliseconds*/)
    {
        std::unique_lock<std::mutex> lock(mutex);

        bool stillFull = !condNonFull.wait_for(lock,
            std::chrono::milliseconds(timeout),
            [this]() { return queue.size() < maxSize; });

        if (stillFull) return false;

        queue.push(item);
        condNonEmpty.notify_all();

        return true;
    }

    /**
     * \brief Attemps to pop the oldest value from the queue.
     *
     * \param [out] item         The oldest value of the queue. It will not be assinged if the queue is empty.
     * \param [in]  timeout      A maximum timeout in ms that the method should be waiting while the queue is empty.
     *
     * \return Status of the operation. Returns `true` in case of the oldest element has been popped from the queue.
     * Otherwise the method returns `false`.
     */
    bool pop(T& item, unsigned int timeout = 1 /*milliseconds*/)
    {
        std::unique_lock<std::mutex> lock(mutex);

        bool stillEmpty = !condNonEmpty.wait_for(lock,
            std::chrono::milliseconds(timeout),
            [this]() { return !queue.empty(); });

        if (stillEmpty) return false;

        item = queue.front();
        queue.pop();
        condNonFull.notify_all();

        return true;
    }

    /**
     * \brief Removes all elements from the queue.
     */
    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex);

        while(!queue.empty())
            queue.pop();

        condNonFull.notify_all();
    }

protected:
    std::queue<T> queue;
    std::size_t maxSize;

    std::mutex mutex;
    std::condition_variable condNonEmpty, condNonFull;

    ThreadSafeQueue(const ThreadSafeQueue&);
    ThreadSafeQueue& operator =(const ThreadSafeQueue&);
};

}
#endif // NVXIO_THREADSAFEQUEUE_HPP
