// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/randomizer.h"

#include <curand.h>
#include <curand_kernel.h>

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
  GPUBackend::Delete(states_, sizeof(curandState) * len_, true);
}

}  // namespace ndll

