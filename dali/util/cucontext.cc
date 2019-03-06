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

CUContext::CUContext(CUdevice device, unsigned int flags)
    : device_{device}, context_{0}, initialized_{false} {
    DALI_ENFORCE(cuInitChecked(),
        "Failed to load libcuda.so. "
        "Check your library paths and if the driver is installed correctly.");
    CUDA_CALL(cuDevicePrimaryCtxRetain(&context_, device));
    push();
    CUdevice dev;
    CUDA_CALL(cuCtxGetDevice(&dev));
    initialized_ = true;
    CUDA_CALL(cuCtxSynchronize());
}

CUContext::~CUContext() {
    if (initialized_) {
        // cuCtxPopCurrent?
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

void CUContext::push() const {
    CUcontext current;
    CUDA_CALL(cuCtxGetCurrent(&current));
    if (current != context_) {
        CUDA_CALL(cuCtxPushCurrent(context_));
    }
}

bool CUContext::initialized() const {
    return initialized_;
}

CUContext::operator CUcontext() const {
    return context_;
}

}  // namespace dali
