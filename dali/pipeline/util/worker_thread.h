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

#ifndef DALI_PIPELINE_UTIL_WORKER_THREAD_H_
#define DALI_PIPELINE_UTIL_WORKER_THREAD_H_

#include <functional>
#include <memory>
#include <string>

#include "dali/core/common.h"

namespace dali {

class DLL_PUBLIC WorkerThread {
 public:
  typedef std::function<void(void)> Work;

  virtual ~WorkerThread() = default;

  /*
   * Separate Shutdown function is needed as we cannot rely on destructor to stop the thread.
   * When the destructor is called other things that work() is using may have been gone long
   * before causing a hang. Now when Shutdown is called we are sure that all things around
   * still exist.
   */
  virtual void Shutdown() = 0;
  virtual void DoWork(Work work) = 0;
  virtual void WaitForWork(bool rethrow_worker_errors = true) = 0;
  virtual void ForceStop() = 0;
  virtual void CheckForErrors() = 0;
  virtual bool WaitForInit() = 0;
};

DLL_PUBLIC std::unique_ptr<WorkerThread> CreateWorkerThread(
    int device_id, bool set_affinity, const std::string &name);

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_WORKER_THREAD_H_
