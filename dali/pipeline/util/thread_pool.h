// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_UTIL_THREAD_POOL_H_
#define DALI_PIPELINE_UTIL_THREAD_POOL_H_

#include <cstdlib>
#include <utility>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <string>
#include "dali/core/common.h"


namespace dali {

class DLL_PUBLIC ThreadPool {
 public:
  // Basic unit of work that our threads do
  typedef std::function<void(int)> Work;

  DLL_PUBLIC ThreadPool(int num_thread, int device_id, bool set_affinity);

  DLL_PUBLIC ~ThreadPool();

  /**
   * @brief Adds work to the queue with optional priority, and optionally starts processing
   *
   * The jobs are queued but the workers don't pick up the work unless they have
   * already been started by a previous call to AddWork with start_immediately = true or RunAll.
   * Once work is started, the threads will continue to pick up whatever work is scheduled
   * until WaitForWork is called.
   */
  DLL_PUBLIC void AddWork(Work work, int64_t priority = 0, bool start_immediately = false);

  /**
   * @brief Wakes up all the threads to complete all the queued work,
   *        optionally not waiting for the work to be finished before return
   *        (the default wait=true is equivalent to invoking WaitForWork after RunAll).
   */
  DLL_PUBLIC void RunAll(bool wait = true);

  /**
   * @brief Waits until all work issued to the thread pool is complete
   */
  DLL_PUBLIC void WaitForWork(bool checkForErrors = true);

  DLL_PUBLIC int NumThreads() const;

  DLL_PUBLIC std::vector<std::thread::id> GetThreadIds() const;

  DISABLE_COPY_MOVE_ASSIGN(ThreadPool);

 private:
  DLL_PUBLIC void ThreadMain(int thread_id, int device_id, bool set_affinity);

  vector<std::thread> threads_;

  using PrioritizedWork = std::pair<int64_t, Work>;
  struct SortByPriority {
    bool operator() (const PrioritizedWork &a, const PrioritizedWork &b) {
      return a.first < b.first;
    }
  };
  std::priority_queue<PrioritizedWork, std::vector<PrioritizedWork>, SortByPriority> work_queue_;

  bool running_;
  bool work_complete_;
  bool started_;
  int active_threads_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::condition_variable completed_;

  //  Stored error strings for each thread
  vector<std::queue<string>> tl_errors_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_THREAD_POOL_H_
