// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_RANDOM_RANDOM_DIST_H_
#define DALI_OPERATORS_RANDOM_RANDOM_DIST_H_

#include <cassert>
#include <random>
#include <type_traits>
#include <utility>
#include "dali/core/math_util.h"
#include "dali/core/cuda_utils.h"
#include "dali/core/host_dev.h"
#include "dali/core/geom/vec.h"
#ifdef __CUDACC__
#include <curand_kernel.h>
#endif

namespace dali {
namespace random {

template <typename CurandState>
struct CurandGenerator {
    CurandState &state;
    __device__ explicit CurandGenerator(CurandState &s) : state(s) {}
    __device__ inline uint32_t operator()() const {
        return curand(&state);
    }
};

template <typename T>
struct normalized_uniform_dist {
  template <typename RNG>
  DALI_HOST_DEV T operator()(RNG &rng) const;
};

DALI_NO_EXEC_CHECK
template <typename RNG>
DALI_HOST_DEV DALI_FORCEINLINE uint32_t get_uint32(RNG &rng) {
  auto x = rng();
  static_assert(sizeof(x) >= 4);  // just throw out higher bits, if any
  static_assert(std::is_integral_v<decltype(x)>);
  return uint32_t(x);
}

DALI_NO_EXEC_CHECK
template <typename RNG>
DALI_HOST_DEV DALI_FORCEINLINE uint64_t get_uint64(RNG &rng) {
  auto x = rng();
  static_assert(sizeof(x) == 4 || sizeof(x) == 8);
  static_assert(std::is_integral_v<decltype(x)>);
  if constexpr (sizeof(x) == 4) {
    uint64_t y = rng();
    return x | (y << 32);
  } else {  // 8
    return x;
  }
}

DALI_NO_EXEC_CHECK
template <typename T, typename RNG>
DALI_HOST_DEV DALI_FORCEINLINE T get_next(RNG &rng) {
  static_assert(sizeof(T) <= 8);
  static_assert(std::is_integral_v<T>);
  if constexpr (sizeof(T) <= 4) {
    return T(get_uint32(rng));
  } else {  // more than 4 bytes
    return T(get_uint64(rng));
  }
}

template <>
template <typename RNG>
DALI_HOST_DEV inline float normalized_uniform_dist<float>::operator()(RNG &rng) const {
  uint32_t r = get_uint32(rng);
  return r * 0x1p-32f;
}

template <>
template <typename RNG>
DALI_HOST_DEV inline double normalized_uniform_dist<double>::operator()(RNG &rng) const {
  uint64_t r = get_uint64(rng);
  return r * 0x1p-64;
}

/** Normal distribution, using Box-Muller transform */
template <typename T>
struct standard_normal_dist {
  template <typename RNG>
  DALI_HOST_DEV inline T operator()(RNG &rng) {
    return get_coord(rng);
  }

 private:
  template <typename RNG>
  DALI_HOST_DEV inline T get_coord(RNG &rng) {
    if (has_box_muller_y) {
      has_box_muller_y = false;
      return box_muller_y;
    }
    T u1 = normalized_uniform_dist<T>()(rng);
    T u2 = normalized_uniform_dist<T>()(rng);

    // Handle zero values - avoid drawing new numbers to avoid desynchronizing the generators
    // across threads in a warp.
    if (u1 == 0) {
      if (u2 == 0) {  // two zeros - the likelihood is 2^-64 for float and 2^-128 for double
        // we can do whatever here, it won't skew the result due to infinitesimal probability
        u1 = T(1e-30f);
      } else {
        // if we get one zero, swap the values instead of generating a new one
        T tmp = u2;
        u2 = u1;
        u1 = tmp;
      }
    }
  #ifdef __CUDA_ARCH__
    T r = sqrt(-2 * log(u1));
    T theta = T(M_PI * 2) * u2;
    T x = r * cos(theta);
    T y = r * sin(theta);
  #else
    T r = std::sqrt(-2 * std::log(u1));
    T theta = T(M_PI * 2) * u2;
    T x = r * std::cos(theta);
    T y = r * std::sin(theta);
  #endif
    has_box_muller_y = true;
    box_muller_y = y;
    return x;
  }

  T box_muller_y = 0;
  bool has_box_muller_y = false;
};

template <typename T>
struct normal_dist {
  T mean = 0, stddev = 1;
  DALI_HOST_DEV normal_dist(T mean, T stddev) : mean(mean), stddev(stddev) {}

  template <typename RNG>
  DALI_HOST_DEV T operator()(RNG &rng) {
    #ifdef __CUDA_ARCH__
      return fma(dist_(rng), stddev, mean);
    #else
      return std::fma(dist_(rng), stddev, mean);
    #endif
  }

 private:
  standard_normal_dist<T> dist_;
};


template <typename T>
struct uniform_real_dist {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
    "Unexpected data type");

  DALI_HOST_DEV uniform_real_dist(const uniform_real_dist &other) = default;
  DALI_HOST_DEV uniform_real_dist &operator=(const uniform_real_dist &other) = default;
  DALI_HOST_DEV uniform_real_dist() = default;
  DALI_HOST_DEV uniform_real_dist(T start, T end) {
    min_value_ = start;
  #ifdef __CUDA_ARCH__
    max_value_ = nextafter(end, start);
  #else
    max_value_ = std::nextafter(end, start);
  #endif
    if (min_value_ > max_value_) {
      cuda_swap(min_value_, max_value_);
    }
    if constexpr (std::is_same_v<T, double>) {
      factor_ = (max_value_ - min_value_) * 0x1p-64;
    } else {
      factor_ = (max_value_ - min_value_) * 0x1p-32f;
    }
  }

  template <typename RNG>
  DALI_HOST_DEV DALI_FORCEINLINE T operator()(RNG &rng) const {
    T val = 0;
    using U = typename std::conditional_t<std::is_same_v<T, double>, uint64_t, uint32_t>;
    U r = get_next<U>(rng);
  #ifdef __CUDA_ARCH__
    val = fma(T(r), factor_, min_value_);
  #else
    val = std::fma(T(r), factor_, min_value_);
  #endif
    // This may lead to slight overrepresentation of the max value but should be negligible.
    val = cuda_min(val, max_value_);
    return val;
  }

 private:
  T min_value_ = 0;
  T max_value_ = std::is_same_v<T, double> ? 0x0.fffffffffffff8p0 : 0x0.ffffffp0f;
  T factor_ = std::is_same_v<T, double> ? 0x1p-64 : 0x1p-32f;
};

/** Uniform distribution over a range of integers.
 *
 * @tparam T The result type. Must be an integral type up to 32 bits.
 *
 * The values are sampled uniformly from the range [start, end) if exclusive_max is false, or
 * [start, end] if exclusive_max is true.
 *
 * For exclusive_max = false, the entire range of the type T may be specified.
 *
 * For large ranges, some values may be sampled more frequently than others due to the limited
 * precision of the random number generator. The mean and variance are not affected, though.
 */
template <typename T = int>
struct uniform_int_dist {
  static_assert(std::is_integral_v<T>, "uniform_int_dist only supports integral types");
  static_assert(sizeof(T) <= 4, "uniform_int_dist only supports types up to 32 bits");

  DALI_HOST_DEV uniform_int_dist(T start, T end, bool exclusive_max = false) {
    assert(exclusive_max ? (end > start) : (end >= start));
    range_start_ = start;
    range_size_ = uint32_t(end) - uint32_t(start) + (exclusive_max ? 0 : 1);
    // NOTE: range_size_ of 0 is valid and denotes the full 32-bit range
  }

  template <typename RNG>
  DALI_HOST_DEV DALI_FORCEINLINE T operator()(RNG &rng) const {
    if ((range_size_ & (range_size_ - 1)) == 0) {  // special case for zero and pow2 range sizes
      // Range mask will be 0xffffffff for range_size_ == 0 due to overflow.
      return range_start_ + (get_uint32(rng) & (range_size_ - 1));
    }
  #ifdef __CUDA_ARCH__
    uint32_t u = get_uint32(rng);
    int x = __umulhi(u, range_size_);
  #else
    uint64_t u = get_uint32(rng);
    int x = (u * range_size_) >> 32;
  #endif
    return range_start_ + x;
  }

 private:
  T range_start_;
  uint32_t range_size_;
};

/** A distribution that samples values from a discrete set.
 *
 * @tparam T The type of the values to sample from.
 *
 * @note For a large number of values, some values may be sampled more frequently than others due
 *       to the limited precision of the random number generator.
 *       There are 2^32 rng outputs assigned to nvalues bins. The probablitties are evenly
 *       distributed only for nvalues much smaller than 2^32 and powers of 2.
 */
template <typename T>
struct uniform_discrete_dist {
 public:
  DALI_HOST_DEV uniform_discrete_dist() : values_(nullptr), nvalues_(0) {
  }

  DALI_HOST_DEV uniform_discrete_dist(const T *values, int64_t nvalues)
    : values_(values), nvalues_(nvalues) {}

  template <typename RNG>
  DALI_HOST_DEV DALI_FORCEINLINE T operator()(RNG &rng) const {
  #ifdef __CUDA_ARCH__
    uint32_t u = get_uint32(rng);
    unsigned idx = __umulhi(u, nvalues_);
  #else
    uint64_t u = get_uint32(rng);
    unsigned idx = (u * nvalues_) >> 32;
  #endif
    return values_[idx];
  }

 private:
  const T *values_ = nullptr;  // device mem pointer
  int64_t nvalues_ = 0;
};

struct bernoulli_dist {
 public:
  DALI_HOST_DEV DALI_FORCEINLINE bernoulli_dist() : threshold(0x7fffffff) {}
  explicit DALI_HOST_DEV DALI_FORCEINLINE bernoulli_dist(float probability) {
    float th = probability * 0x1p32f;
    if (th >= 0x1p32f) {  // avoid overflow
      threshold = 0xffffffff;
    } else {
      threshold = static_cast<uint32_t>(th);
    }
  }

  template <typename RNG>
  DALI_HOST_DEV DALI_FORCEINLINE bool operator()(RNG &rng) const {
    return get_uint32(rng) <= threshold;
  }

 private:
  uint32_t threshold = 0x7fffffff;
};


/** Poisson distribution, using either curand_poisson or std::poisson_distribution.
 *
 * NOTE: Results are different between CPU and GPU variant.
 *
 * TODO(michalz): Add an implementation that is consistent between CPU and GPU.
 */
struct poisson_dist {
  float mean = 1.0f;
  DALI_HOST_DEV poisson_dist() = default;
  DALI_HOST_DEV explicit poisson_dist(float mean) : mean(mean) {}

#if defined(__clang__) && defined(__CUDA__)
  // Workaround for a clang-only build for cases where operator() is used in a host-device function
  template <typename State>
  DALI_HOST_DEV DALI_FORCEINLINE uint32_t operator()(CurandGenerator<State> &rng) const {
    #ifdef __CUDA_ARCH__
      return curand_poisson(&rng.state, mean);
    #else
      assert(!"Unreachable code!");
      return 0;
    #endif
  }

  template <typename RNG>
  __host__ DALI_FORCEINLINE uint32_t operator()(RNG &rng) const {
    #ifndef __CUDA_ARCH__
      return std::poisson_distribution<uint32_t>(mean)(rng);
    #else
      assert(!"Unreachable code!");
      return 0;
    #endif
  }
#else
  template <typename State>
  __device__ DALI_FORCEINLINE uint32_t operator()(CurandGenerator<State> &rng) const {
    return curand_poisson(&rng.state, mean);
  }

  template <typename RNG>
  __host__ DALI_FORCEINLINE uint32_t operator()(RNG &rng) const {
    return std::poisson_distribution<uint32_t>(mean)(rng);
  }
#endif
};

}  // namespace random
}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_RANDOM_DIST_H_
