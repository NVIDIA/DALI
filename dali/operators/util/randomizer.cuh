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
#include "dali/kernels/alloc.h"
#include "dali/pipeline/data/backend.h"
#include <curand_kernel.h>  // NOLINT

namespace dali {

struct curand_states {
  curand_states(uint64_t seed, size_t len);
  ~curand_states();

  DALI_HOST_DEV inline curandState* states() {
    return states_;
  }

  DALI_HOST_DEV inline curandState& operator[](size_t idx) {
    assert(idx < len_);
    return states_[idx];
  }

 private:
  size_t len_;
  int device_;
  kernels::memory::KernelUniquePtr<curandState> states_mem_;
  curandState* states_;  // std::unique_ptr::get can't be called from __device__ functions
};

template <typename T>
struct curand_normal_dist {};

template <>
struct curand_normal_dist<float> {
  float mean = 0.0f, stddev = 1.0f;

  __device__ inline float yield(curandState *state) {
    return mean + curand_normal(state) * stddev;
  }
};

template <>
struct curand_normal_dist<double> {
  double mean = 0.0f, stddev = 1.0f;

  __device__ inline double yield(curandState *state) {
    return mean + curand_normal_double(state) * stddev;
  }
};

template <typename T>
struct curand_uniform_dist {};

template <>
struct curand_uniform_dist<float> {
  DALI_HOST_DEV curand_uniform_dist(float start, float end)
    : range_start_(start), range_size_(end-start) {}

  DALI_HOST_DEV curand_uniform_dist()
    : range_start_(-1.0f), range_size_(2.0f) {}

  __device__ inline float yield(curandState *state) {
    return range_start_ + curand_uniform(state) * range_size_;
  }
 private:
  float range_start_, range_size_;
};

template <>
struct curand_uniform_dist<double> {
  DALI_HOST_DEV curand_uniform_dist(float start, float end)
    : range_start_(start), range_size_(end-start) {}

  DALI_HOST_DEV curand_uniform_dist()
    : range_start_(-1.0f), range_size_(2.0f) {}

  __device__ inline double yield(curandState *state) {
    return range_start_ + curand_uniform_double(state) * range_size_;
  }

 private:
  double range_start_, range_size_;
};

struct curand_uniform_int_range_dist {
  DALI_HOST_DEV curand_uniform_int_range_dist(int start, int end)
    : range_start_(start), range_size_(end-start) {}
  
  __device__ inline int yield(curandState *state) {
    return range_start_ + (curand(state) % range_size_);
  }

 private:
  int range_start_;
  unsigned int range_size_;
};

template <typename T>
struct curand_uniform_int_values_dist {
  const T *values = nullptr;  // device mem pointer
  int64_t nvalues = 0;

  __device__ inline double yield(curandState *state) {
    return values[curand(state) % nvalues];
  }
};

}  // namespace dali

#endif  // DALI_OPERATORS_UTIL_RANDOMIZER_CUH_
