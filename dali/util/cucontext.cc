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

#include "dali/util/cuda_utils.h"
#include "dali/util/cucontext.h"
#include "dali/util/dynlink_cuda.h"


namespace dali {

CUContext::CUContext() : context_{0}, initialized_{false} {
}

CUContext::CUContext(int device_id, unsigned int flags)
    : device_{0}, device_id_{-1}, context_{0}, initialized_{false} {
  // for invalid device create empty context
  if (device_id == -1) {
    return;
  }
  DALI_ENFORCE(cuInitChecked(),
    "Failed to load libcuda.so. "
    "Check your library paths and if the driver is installed correctly.");
  device_id_ = device_id;
  CUDA_CALL(cuDeviceGet(&device_, device_id_));
  CUDA_CALL(cuDevicePrimaryCtxRetain(&context_, device_));
  initialized_ = true;
  bool revert = push();
  CUDA_CALL(cuCtxSynchronize());
  if (revert) {
    pop();
  }
}

CUContext::~CUContext() {
  if (initialized_) {
    CUDA_CALL(cuDevicePrimaryCtxRelease(device_));
  }
}

CUContext::CUContext(CUContext&& other)
  : device_{other.device_}, context_{other.context_},
    initialized_{other.initialized_} {
  other.device_ = 0;
  other.context_ = 0;
  other.initialized_ = false;
}

CUContext& CUContext::operator=(CUContext&& other) {
  if (initialized_) {
    CUDA_CALL(cuCtxDestroy(context_));
  }
  device_ = other.device_;
  context_ = other.context_;
  initialized_ = other.initialized_;
  other.device_ = 0;
  other.context_ = 0;
  other.initialized_ = false;
  return *this;
}

bool CUContext::push() const {
  CUcontext current;
  CUDA_CALL(cuCtxGetCurrent(&current));
  if (current != context_) {
    CUDA_CALL(cuCtxPushCurrent(context_));
    if  (current != NULL) {
      return true;
    }
  } 
  return false;
}


void CUContext::pop() const {
  CUcontext new_ctx;
  CUDA_CALL(cuCtxPopCurrent(&new_ctx));
}

CUContext::operator CUcontext() const {
  return context_;
}

}  // namespace dali
