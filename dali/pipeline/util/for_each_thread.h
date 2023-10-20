// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_PIPELINE_UTIL_FOR_EACH_THREAD_H_
#define DALI_PIPELINE_UTIL_FOR_EACH_THREAD_H_

#include <mutex>
#include <condition_variable>
#include <atomic>

#include "dali/pipeline/util/thread_pool.h"

namespace dali {

/**
 * @brief Executes a function `func` once in each thread in the thread pool
 *
 * The function schedules a callable `func` and adds synchronization that guarantees
 * that the function is executed exactly once in each of the thread pool's threads.
 * The function waits until all callables have completed.
 *
 * @param tp    a thread pool object
 * @param func  a callable, taking one integer parameter (thread index within the thread pool)
 *
 * @note This function is quite expensive - it creates a temporary mutex and condition variable.
 *       The intended usage is for initialization and shutdown of thread local storage or
 *       other thread-bound resources.
 */
template <typename Func>
void ForEachThread(ThreadPool &tp, Func &&func) {
  std::mutex m;
  std::condition_variable cv;
  int n = tp.NumThreads();
  std::atomic_int pending{n};
  for (int i = 0; i < n; i++) {
    tp.AddWork([&](int tid) {
      std::exception_ptr err{nullptr};
      try {
        func(tid);
      } catch (...) {
        // If there's any error we catch it and rethrow only after the number
        // of pending tasks is properly updated and the threads unblocked.
        err = std::current_exception();
      }

      std::unique_lock lock(m);
      if (--pending == 0) {
        cv.notify_all();
      } else {
        cv.wait(lock, [&]() { return pending == 0; });
      }
      if (err)  // if there's an error, we can rethrow it now
        std::rethrow_exception(err);
    });
  }
  tp.RunAll(true);
}

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_FOR_EACH_THREAD_H_
