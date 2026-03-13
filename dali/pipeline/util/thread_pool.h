// Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <atomic>
#include <cstdlib>
#include <utility>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <stdexcept>
#include <string>
#include "dali/core/common.h"
#if NVML_ENABLED
#include "dali/util/nvml.h"
#endif
#include "dali/core/semaphore.h"
#include "dali/core/spinlock.h"
#include "dali/core/exec/thread_idx.h"
#include "dali/pipeline/util/thread_pool_interface.h"

namespace dali {

class DLL_PUBLIC OldThreadPool : public ThreadPool {
 public:
  using Work = std::function<void()>;
  using WorkWithThreadIdx = std::function<void(int)>;

  OldThreadPool(int num_thread, int device_id, bool set_affinity, const char* name);

  OldThreadPool(int num_thread, int device_id, bool set_affinity, const std::string& name)
      : OldThreadPool(num_thread, device_id, set_affinity, name.c_str()) {}

  ~OldThreadPool();

  /**
   * @brief Adds work to the queue with optional priority, and optionally starts processing
   *
   * The jobs are queued but the workers don't pick up the work unless they have
   * already been started by a call to RunAll.
   * Once work is started, the threads will continue to pick up whatever work is scheduled
   * until WaitForWork is called.
   */
  void AddWork(WorkWithThreadIdx work, int64_t priority = 0) override;

  void AddWork(Work work, int64_t priority = 0) override;

  /**
   * @brief Wakes up all the threads to complete all the queued work,
   *        optionally not waiting for the work to be finished before return
   *        (the default wait=true is equivalent to invoking WaitForWork after RunAll).
   */
  void RunAll(bool wait = true) override;

  /**
   * @brief Waits until all work issued to the thread pool is complete
   */
  void WaitForWork() override;

  int NumThreads() const override;

  std::vector<std::thread::id> GetThreadIds() const override;

  DISABLE_COPY_MOVE_ASSIGN(OldThreadPool);

 private:
  void WaitForWork(bool checkForErrors);

  void ThreadMain(int thread_id, int device_id, bool set_affinity, const std::string &name);

  vector<std::thread> threads_;

  using PrioritizedWork = std::pair<int64_t, std::function<void(int)>>;
  struct SortByPriority {
    bool operator() (const PrioritizedWork &a, const PrioritizedWork &b) {
      return a.first < b.first;
    }
  };
  std::priority_queue<PrioritizedWork, std::vector<PrioritizedWork>, SortByPriority> work_queue_;

  alignas(64) spinlock queue_lock_;
  dali::counting_semaphore queue_semaphore_{0};
  bool running_ = true;
  bool started_ = false;
  alignas(64) std::atomic_int outstanding_work_{0};
  std::mutex completed_mutex_;
  std::condition_variable completed_;

  // Stored errors for each thread
  vector<std::queue<std::exception_ptr>> tl_errors_;
#if NVML_ENABLED
  nvml::NvmlInstance nvml_handle_;
#endif
};

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_THREAD_POOL_H_
