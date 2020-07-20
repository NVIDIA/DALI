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


#ifndef DALI_OPERATORS_UTIL_RANDOMIZER_CUH_
#define DALI_OPERATORS_UTIL_RANDOMIZER_CUH_

#include <math.h>
#include "dali/core/device_guard.h"
#include "dali/pipeline/data/backend.h"
#include <curand_kernel.h>  // NOLINT

namespace dali {

namespace detail {

template <typename T>
struct CurandNormal {};

template <>
struct CurandNormal<float> {
  __device__ static float normal(curandState *state) {
    return curand_normal(state);
  }
};

template <>
struct CurandNormal<double> {
  __device__ static double normal(curandState *state) {
    return curand_normal_double(state);
  }
};

}  // namespace detail

class RandomizerGPU {
 public:
  explicit RandomizerGPU(int seed, size_t len);

  __device__ inline int rand(int idx) {
    return curand(&states_[idx]);
  }

  template <typename T>
  __device__ inline T normal(int idx) {
    return detail::CurandNormal<T>::normal(&states_[idx]);
  }

  void Cleanup();

 private:
    curandState* states_;
    size_t len_;
    int device_;
    static constexpr int block_size_ = 256;
};

}  // namespace dali

#endif  // DALI_OPERATORS_UTIL_RANDOMIZER_CUH_
