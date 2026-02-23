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

class DLL_PUBLIC ThreadPoolFacade : public ThreadPool {
 public:
  explicit ThreadPoolFacade(ThreadPoolBase *thread_pool) : tp_(thread_pool) {}
  ~ThreadPoolFacade() override;

  void AddWork(std::function<void()> work, int64_t priority = 0) override;
  void AddWork(std::function<void(int)> work, int64_t priority = 0) override;
  void RunAll(bool wait = true) override;
  void WaitForWork() override;
  int NumThreads() const;
  std::vector<std::thread::id> GetThreadIds() const;

 private:
  ThreadPoolBase *tp_ = nullptr;
  std::optional<Job> job_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_NEW_THREAD_POOL_H_
