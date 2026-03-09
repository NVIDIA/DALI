// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_UTIL_NEW_THREAD_POOL_H_
#define DALI_PIPELINE_UTIL_NEW_THREAD_POOL_H_

#include <list>
#include <optional>
#include <string>
#include <vector>
#include "dali/core/exec/thread_pool_base.h"
#if NVML_ENABLED
#include "dali/util/nvml.h"
#endif
#include "dali/pipeline/util/thread_pool_interface.h"

namespace dali {

class DLL_PUBLIC NewThreadPool : public ThreadPoolBase {
 public:
  NewThreadPool(int num_threads, std::optional<int> device_id, bool set_affinity, std::string name);

 private:
  std::any OnThreadStart(int thread_idx, bool set_affinity);
  std::optional<int> device_id_;
  std::string name_;
#if NVML_ENABLED
  nvml::NvmlInstance nvml_handle_;
#endif
};

/** Combines an existing ThreadPoolBase and Jobs to provide a backward-compatible interface.
 *
 * This class wraps a (non-owning) pointer to a ThreadPoolBase object and an owning list of Jobs.
 * The jobs are created as needed when AddWork is called and destroyed when Wait is called.
 *
 * There's a subtle difference in priority handling between old thread pool and this wrapper
 * when adding more work while the previous work is already started.
 */
class DLL_PUBLIC ThreadPoolFacade : public ThreadPool {
 public:
  /** Constructs a ThreadPool facade for a ThreadPoolBase object
   *
   * @param thread_pool   A pointer to an existing thread pool. The caller must keep it alive
   *                      until all work items submitted to the facade have been waited for.
   */
  explicit ThreadPoolFacade(ThreadPoolBase *thread_pool) : tp_(thread_pool) {}
  ~ThreadPoolFacade() noexcept override;

  /** Adds a new wokr item, with a priority. Higher priority items are picked up first.
   *
   *  After work has been added, it must be run and waited for.
   *
   *  @param work A parameterless function representing the work item.
   */
  void AddWork(std::function<void()> work, int64_t priority = 0) override;
  /** Adds a new wokr item, with a priority. Higher priority items are picked up first.
   *
   *  After work has been added, it must be run and waited for.
   *
   *  @param work A function representing the work item, parameterized with a thread index.
   */
  void AddWork(std::function<void(int)> work, int64_t priority = 0) override;

  /** Sumbits all work added for execution.
   *
   * Adding more work after this call requires calling RunAll again.
   */
  void RunAll(bool wait = true) override;

  /** Waits for the work to complete.
   *
   * This function waits until all work items are complete.
   * If any of them throws an exception, the function will rethrow it.
   * If multiple items throw, the exceptions are wrapped into MultiplErrors exception.
   */
  void WaitForWork() override;

  /** Returns the number of threads in the underlying thread pool */
  int NumThreads() const override;

  /** Returns the systsm-specific thread ids (not indices) of the threads in the thread pool */
  std::vector<std::thread::id> GetThreadIds() const override;

 private:
  ThreadPoolBase *tp_ = nullptr;
  std::list<Job> jobs_;
};

DLL_PUBLIC bool UseNewThreadPool();

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_NEW_THREAD_POOL_H_
