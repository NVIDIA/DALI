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

#include <atomic>
#include <condition_variable>
#include <mutex>
#include "dali/dali.h"
#include "dali/c_api_2/error_handling.h"
#include "dali/c_api_2/pipeline_registry.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/init.h"
#include "dali/pipeline/operator/op_spec.h"

using namespace dali;  // NOLINT

namespace {
std::atomic<int> g_init_count;
std::atomic<bool> g_was_initialized;
// Serializes daliInit/daliShutdown, so the init count and the registry state never diverge.
std::mutex g_lifecycle_mtx;

std::atomic<int> g_active_calls;
std::mutex g_active_calls_mtx;
std::condition_variable g_active_calls_cv;

void WaitForActiveCalls() {
  std::unique_lock lock(g_active_calls_mtx);
  g_active_calls_cv.wait(lock, [] { return g_active_calls.load() == 0; });
}
}  // namespace

namespace dali::c_api {
  daliResult_t CheckInit() {
    if (g_init_count <= 0) {
      if (g_was_initialized)
        return DALI_ERROR_UNLOADING;
      else
        return daliInit();
    }
    return DALI_SUCCESS;
  }

  ActiveCallGuard::ActiveCallGuard() {
    g_active_calls++;
  }

  ActiveCallGuard::~ActiveCallGuard() {
    if (--g_active_calls == 0) {
      std::lock_guard lock(g_active_calls_mtx);
      g_active_calls_cv.notify_all();
    }
  }
}  // namespace dali::c_api

daliResult_t daliInit() {
  try {  // cannot use DALI_PROLOG in this function, since DALI isn't initialized yet
    static int init = []() {
      DALIInit(OpSpec("CPUAllocator"),
               OpSpec("PinnedCPUAllocator"),
               OpSpec("GPUAllocator"));
      return 0;
    }();
    (void)init;
    std::lock_guard lifecycle_lock(g_lifecycle_mtx);
    dali::c_api::PipelineRegistry::instance().Open();
    g_init_count++;
    g_was_initialized = true;
    return DALI_SUCCESS;
  } catch (...) {
    return dali::c_api::HandleError(std::current_exception());
  }
}

daliResult_t daliShutdown() {
  try {  // cannot use DALI_PROLOG - the final shutdown waits for all the other calls to finish
    if (auto err = dali::c_api::CheckInit())
      return err;
    std::lock_guard lifecycle_lock(g_lifecycle_mtx);
    if (g_init_count <= 0)
      return DALI_ERROR_UNLOADING;
    if (--g_init_count == 0) {
      auto &registry = dali::c_api::PipelineRegistry::instance();
      registry.Close();  // reject new pipelines...
      WaitForActiveCalls();  // ...let the calls admitted before closing finish...
      size_t destroyed = registry.DestroyAll();  // ...and destroy whatever is left
      if (destroyed > 0) {
        DALI_WARN(destroyed, " pipeline instance(s) were not destroyed before daliShutdown was "
                  "called. Destroying them now.");
      }
    }
    return DALI_SUCCESS;
  } catch (...) {
    return dali::c_api::HandleError(std::current_exception());
  }
}
