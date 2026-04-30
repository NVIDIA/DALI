// Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/device_guard.h"
#include "dali/core/cuda_error.h"

namespace dali {

// This makes restoring NULL context possible
#define INVALID_CONTEXT ((CUcontext)(intptr_t)-1)  // NOLINT

DeviceGuard::DeviceGuard() :
  old_context_(INVALID_CONTEXT) {
  DALI_ENFORCE(cuInitChecked(),
    "Failed to load libcuda.so. "
    "Check your library paths and if the driver is installed correctly.");
  CUDA_CALL(cuCtxGetCurrent(&old_context_));
}

namespace {

struct PrimaryContext {
  ~PrimaryContext() {
    CUDA_DTOR_CALL(cuDevicePrimaryCtxRelease(device));
  }

  CUdevice device{};
  std::atomic<CUcontext> handle{nullptr};

  CUcontext Get() {
    CUcontext ctx = handle.load();
    if (ctx)
      return ctx;
    CUDA_CALL(cuDevicePrimaryCtxRetain(&ctx, device));
    CUcontext expected = nullptr;
    if (handle.compare_exchange_strong(expected, ctx)) {
      return ctx;
    } else {
      CUDA_CALL(cuDevicePrimaryCtxRelease(device));
      return expected;
    }
  }
};



}  // namespace

DeviceGuard::DeviceGuard(int new_device) :
  old_context_(INVALID_CONTEXT) {
  if (new_device >= 0) {
    DALI_ENFORCE(cuInitChecked(),
      "Failed to load libcuda.so. "
      "Check your library paths and if the driver is installed correctly.");
    CUDA_CALL(cuCtxGetCurrent(&old_context_));
    static auto default_contexts = []() {
      int ndevs = 0;
      CUDA_CALL(cudaGetDeviceCount(&ndevs));
      std::vector<PrimaryContext> ctxs(ndevs);
      for (int i = 0; i < ndevs; i++)
        ctxs[i].device = i;
      return ctxs;
    }();
    if (new_device >= static_cast<int>(default_contexts.size()))
      throw std::out_of_range(make_string("Invalid device ordinal: ", new_device));
    auto *ctx = default_contexts[new_device].Get();
    CUDA_CALL(cuCtxSetCurrent(ctx));
  }
}

DeviceGuard::~DeviceGuard() {
  if (old_context_ != INVALID_CONTEXT) {
    CUresult err = cuCtxSetCurrent(old_context_);
    if (err != CUDA_SUCCESS) {
      std::cerr << "Failed to recover from DeviceGuard: " << err << std::endl;
      std::terminate();
    }
  }
}

}  // namespace dali
