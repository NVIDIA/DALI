// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#define DALI_ALLOW_NEW_C_API
#include "dali/dali.h"
#include "dali/c_api_2/error_handling.h"
#include "dali/pipeline/init.h"
#include "dali/pipeline/operator/op_spec.h"

using namespace dali;  // NOLINT

namespace {
std::atomic<int> g_init_count;
std::atomic<bool> g_was_initialized;
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
    g_init_count++;
    g_was_initialized = true;
    return DALI_SUCCESS;
  } catch (...) {
    return dali::c_api::HandleError(std::current_exception());
  }
}

daliResult_t daliShutdown() {
  DALI_PROLOG();
  int init_count = --g_init_count;
  if (init_count < 0) {
    ++g_init_count;
    return DALI_ERROR_UNLOADING;
  }
  if (init_count == 0) {
    // actual shutdown code goes here
  }
  DALI_EPILOG();
}
