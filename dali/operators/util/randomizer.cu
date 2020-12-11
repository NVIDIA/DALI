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

#include "dali/operators/util/randomizer.cuh"

namespace dali {

namespace detail {

__global__
void init_states(const size_t N, uint64_t seed, curandState *states) {
  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
       idx < N;
       idx += blockDim.x * gridDim.x) {
    curand_init(seed, idx, 0, &states[idx]);
  }
}

}  // namespace detail

curand_states::curand_states(uint64_t seed, size_t len) : len_(len) {
  cudaGetDevice(&device_);
  states_mem_ = kernels::memory::alloc_unique<curandState>(kernels::AllocType::GPU, len);
  states_ = states_mem_.get();
  static constexpr int kBlockSize = 256;
  int grid = div_ceil(len_, kBlockSize);
  detail::init_states<<<grid, kBlockSize>>>(len_, seed, states_);
}

}  // namespace dali
