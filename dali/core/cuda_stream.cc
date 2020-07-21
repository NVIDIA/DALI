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

#include "dali/core/cuda_stream.h"
#include "dali/core/cuda_utils.h"
#include "dali/core/device_guard.h"

namespace dali {

CUDAStream CUDAStream::Create(bool non_blocking, int device_id) {
  cudaStream_t stream;
  int flags = non_blocking ? cudaStreamNonBlocking : cudaStreamDefault;
  DeviceGuard dg(device_id);
  CUDA_CALL(cudaStreamCreateWithFlags(&stream, flags));
  return CUDAStream(stream);
}

CUDAStream CUDAStream::CreateWithPriority(bool non_blocking, int priority, int device_id) {
  cudaStream_t stream;
  int flags = non_blocking ? cudaStreamNonBlocking : cudaStreamDefault;
  DeviceGuard dg(device_id);
  CUDA_CALL(cudaStreamCreateWithPriority(&stream, flags, priority));
  return CUDAStream(stream);
}

void CUDAStream::DestroyHandle(cudaStream_t stream) {
  CUDA_DTOR_CALL(cudaStreamDestroy(stream));
}

}  // namespace dali
