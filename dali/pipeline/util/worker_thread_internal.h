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

#ifndef DALI_PIPELINE_UTIL_WORKER_THREAD_INTERNAL_H_
#define DALI_PIPELINE_UTIL_WORKER_THREAD_INTERNAL_H_

#include "dali/pipeline/util/worker_thread.h"

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <string>
#include <string_view>
#include <thread>

#if NVML_ENABLED
#include "dali/util/nvml.h"
#endif

namespace dali {

namespace detail {

class Barrier {
 public:
  explicit Barrier(std::size_t count) : count_(count), current_(count) {}

  void Wait(bool reset = false);
  void Break();

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::size_t count_;
  std::size_t current_;
};

}  // namespace detail

class DLL_PUBLIC WorkerThreadImpl final : public WorkerThread {
 public:
  explicit WorkerThreadImpl(int device_id, bool set_affinity, std::string_view name);
  ~WorkerThreadImpl() override;

  void Shutdown() override;
  void DoWork(Work work) override;
  void WaitForWork(bool rethrow_worker_errors = true) override;
  void ForceStop() override;
  void CheckForErrors() override;
  bool WaitForInit() override;

 private:
  void ThreadMain(int device_id, bool set_affinity, const std::string &name);

  bool running_ = true;
  bool work_complete_ = true;
  std::queue<Work> work_queue_;
  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable cv_, completed_;

  std::queue<std::string> errors_;

  detail::Barrier barrier_{2};
#if NVML_ENABLED
  nvml::NvmlInstance nvml_handle_;
#endif
};

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_WORKER_THREAD_INTERNAL_H_
