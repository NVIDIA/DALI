// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_UTIL_RANDOMIZER_IMPL_GPU_CUH_
#define NDLL_PIPELINE_OPERATORS_UTIL_RANDOMIZER_IMPL_GPU_CUH_

#include "ndll/pipeline/operators/util/randomizer.h"

#include <curand.h>
#include <curand_kernel.h>

#include "ndll/pipeline/util/device_guard.h"

namespace ndll {

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

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_UTIL_RANDOMIZER_IMPL_GPU_CUH_
