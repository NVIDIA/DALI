// Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cassert>
#include <memory>
#include "dali/core/host_dev.h"
#include "dali/core/device_guard.h"
#include "dali/core/mm/memory.h"
#include <curand_kernel.h>  // NOLINT

namespace dali {

struct curand_states {
  curand_states(uint64_t seed, size_t len);

  inline explicit curand_states(size_t len) : len_(len) {
    states_mem_ = mm::alloc_raw_shared<curandState, mm::memory_kind::device>(len);
    states_ = states_mem_.get();
  }

  DALI_HOST_DEV inline curandState* states() {
    return states_;
  }

  DALI_HOST_DEV inline const curandState* states() const {
    return states_;
  }

  DALI_HOST_DEV inline curandState& operator[](size_t idx) {
    assert(idx < len_);
    return states_[idx];
  }

  DALI_HOST inline size_t length() const {
    return len_;
  }

  DALI_HOST inline curand_states copy(AccessOrder order) const {
    curand_states states(len_);
    CUDA_CALL(cudaMemcpyAsync(states.states_, states_, sizeof(curandState) * len_,
                              cudaMemcpyDeviceToDevice, order.stream()));
    return states;
  }

  DALI_HOST inline void set(const curand_states &other) {
    CUDA_CALL(cudaMemcpyAsync(states_, other.states_, sizeof(curandState) * len_,
                              cudaMemcpyDeviceToDevice, cudaStreamDefault));
    CUDA_CALL(cudaStreamSynchronize(cudaStreamDefault));
  }

 private:
  size_t len_;
  std::shared_ptr<curandState> states_mem_;
  curandState* states_;  // std::shared_ptr::get can't be called from __device__ functions
};

template <typename T>
struct curand_normal_dist;

template <>
struct curand_normal_dist<float> {
  float mean = 0.0f, stddev = 1.0f;

  __device__ inline float operator()(curandState *state) const {
    return mean + curand_normal(state) * stddev;
  }
};

template <>
struct curand_normal_dist<double> {
  double mean = 0.0f, stddev = 1.0f;

  __device__ inline double operator()(curandState *state) const {
    return mean + curand_normal_double(state) * stddev;
  }
};

template <typename T>
struct curand_uniform_dist {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
    "Unexpected data type");

  DALI_HOST_DEV curand_uniform_dist() = default;

  DALI_HOST_DEV curand_uniform_dist(T start, T end)
      : range_start_(start), range_end_(end), range_size_(end-start) {
    assert(end > start);
  }

  __device__ inline T operator()(curandState *state) const {
    T val;
    if (std::is_same<T, double>::value) {
      do {
        val = range_start_ + curand_uniform_double(state) * range_size_;
      } while (val >= range_end_);
    } else {
      do {
        val = range_start_ + curand_uniform(state) * range_size_;
      } while (val >= range_end_);
    }
    return val;
  }

 private:
  T range_start_ = 0, range_end_ = 1, range_size_ = 1;
};

struct curand_uniform_int_range_dist {
  DALI_HOST_DEV curand_uniform_int_range_dist(int start, int end)
      : range_start_(start), range_size_(end-start) {
    assert(end > start);
  }

  __device__ inline int operator()(curandState *state) const {
    return range_start_ + (curand(state) % range_size_);
  }

 private:
  int range_start_;
  unsigned int range_size_;
};

template <typename T>
struct curand_uniform_int_values_dist {
 public:
  DALI_HOST_DEV curand_uniform_int_values_dist() : values_(nullptr), nvalues_(0) {
    // Should not be used. It is just here to make the base
    // RNG operator code easier.
    assert(false);
  }
  DALI_HOST_DEV curand_uniform_int_values_dist(const T *values, int64_t nvalues)
    : values_(values), nvalues_(nvalues) {}

  __device__ inline double operator()(curandState *state) const {
    return values_[curand(state) % nvalues_];
  }

 private:
  const T *values_ = nullptr;  // device mem pointer
  int64_t nvalues_ = 0;
};

struct curand_bernoulli_dist {
 public:
  explicit DALI_HOST_DEV curand_bernoulli_dist(float probability = 0.5f)
    : probability_(probability) {}

  __device__ inline bool operator()(curandState *state) const {
    return curand_uniform(state) <= probability_;
  }

 private:
  float probability_ = 0.5f;
};

struct curand_poisson_dist {
 public:
  explicit DALI_HOST_DEV curand_poisson_dist(float lambda)
    : lambda_(lambda) {}

  __device__ inline unsigned int operator()(curandState *state) const {
    return curand_poisson(state, lambda_);
  }

 private:
  float lambda_ = 0.0f;
};

}  // namespace dali

#endif  // DALI_OPERATORS_UTIL_RANDOMIZER_CUH_
