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

#ifndef DALI_PIPELINE_OPERATORS_UTIL_RANDOMIZER_IMPL_GPU_CUH_
#define DALI_PIPELINE_OPERATORS_UTIL_RANDOMIZER_IMPL_GPU_CUH_

#include "dali/pipeline/operators/util/randomizer.h"

#include <curand.h>
#include <curand_kernel.h>

#include "dali/util/device_guard.h"

namespace dali {

__global__
void initializeStates(const int N, unsigned int seed, curandState *states) {
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
       idx < N;
       idx += blockDim.x * gridDim.x) {
    curand_init(seed, idx, 0, &states[idx]);
  }
}

template <>
Randomizer<GPUBackend>::Randomizer(int seed, size_t len) {
  len_ = len;
  cudaGetDevice(&device_);
  states_ = GPUBackend::New(sizeof(curandState) * len, true);
  initializeStates<<<128, 256>>>(len_, seed, reinterpret_cast<curandState*>(states_));
}

template <>
__device__
int Randomizer<GPUBackend>::rand(int idx) {
  return curand(reinterpret_cast<curandState*>(states_) + idx);
}

template <>
void Randomizer<GPUBackend>::Cleanup() {
  DeviceGuard g(device_);
  GPUBackend::Delete(states_, sizeof(curandState) * len_, true);
}

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_UTIL_RANDOMIZER_IMPL_GPU_CUH_
