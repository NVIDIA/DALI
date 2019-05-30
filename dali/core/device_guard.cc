// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/core/device_guard.h"
#include "dali/core/error_handling.h"

namespace dali {

DeviceGuard::DeviceGuard() :
  old_context_(NULL) {
  DALI_ENFORCE(cuInitChecked(),
    "Failed to load libcuda.so. "
    "Check your library paths and if the driver is installed correctly.");
  CUDA_CALL(cuCtxGetCurrent(&old_context_));
}

DeviceGuard::DeviceGuard(int new_device) :
  old_context_(NULL) {
  if (new_device >= 0) {
    DALI_ENFORCE(cuInitChecked(),
      "Failed to load libcuda.so. "
      "Check your library paths and if the driver is installed correctly.");
    CUDA_CALL(cuCtxGetCurrent(&old_context_));
    CUDA_CALL(cudaSetDevice(new_device));
  }
}

DeviceGuard::~DeviceGuard() {
  if (old_context_ != NULL) {
    CUresult err = cuCtxSetCurrent(old_context_);
    if (err != CUDA_SUCCESS) {
      std::cerr << "Failed to recover from DeviceGuard: " << err << std::endl;
      std::terminate();
    }
  }
}

}  // namespace dali
