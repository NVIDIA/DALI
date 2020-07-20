// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/core/cuda_event.h"
#include "dali/core/cuda_utils.h"
#include "dali/core/device_guard.h"

namespace dali {

CUDAEvent CUDAEvent::Create(int device_id) {
  return CreateWithFlags(cudaEventDisableTiming, device_id);
}

CUDAEvent CUDAEvent::CreateWithFlags(unsigned flags, int device_id) {
  cudaEvent_t event;
  DeviceGuard dg(device_id);
  CUDA_CALL(cudaEventCreateWithFlags(&event, flags));
  return CUDAEvent(event);
}

void CUDAEvent::DestroyHandle(cudaEvent_t event) {
  CUDA_DTOR_CALL(cudaEventDestroy(event));
}

}  // namespace dali
