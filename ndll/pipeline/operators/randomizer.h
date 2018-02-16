// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifndef NDLL_PIPELINE_OPERATORS_RANDOMIZER_H_
#define NDLL_PIPELINE_OPERATORS_RANDOMIZER_H_

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

class Randomizer {
 public:
  explicit Randomizer(int seed = 1234, size_t len = 128*32*32) : len_(len) {
        states_ = reinterpret_cast<curandState*>(GPUBackend::New(sizeof(curandState) * len, true));
        initializeStates<<<128, 256>>>(len_, seed, states_);
    }

    __host__ __device__
    int rand(int idx) {
#ifdef __CUDA_ARCH__
        return curand(states_ + idx);
#else
        return lrand48();
#endif
    }

  void Cleanup() {
    GPUBackend::Delete(states_, sizeof(curandState) * len_, true);
  }

 private:
    curandState *states_;
    size_t len_;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_RANDOMIZER_H_
