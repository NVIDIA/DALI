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

#ifndef DALI_CORE_FAST_DIV_H_
#define DALI_CORE_FAST_DIV_H_

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace dali {

namespace detail {

template <typename component>
struct lohi {
  component lo, hi;
};

template <typename T>
__host__ __device__ __forceinline__ lohi<T> operator<<(lohi<T> x, unsigned sh) {
  using U = typename std::make_unsigned<T>::type;
  static constexpr unsigned bits = sizeof(T) * 8;
  if (sh == 0) {
    return x;
  } else if (sh >= bits) {
    return { 0, x.lo << (sh-bits) };
  } else {
    return {
      x.lo << sh,
      x.hi << sh | U(x.lo) >> (bits - sh)
    };
  }
}

template <typename T>
__host__ __device__ __forceinline__ lohi<T> operator>>(lohi<T> x, unsigned sh) {
  using U = typename std::make_unsigned<T>::type;
  static constexpr unsigned bits = sizeof(T) * 8;
  if (sh == 0) {
    return x;
  } else if (sh >= bits) {
    return { x.hi >> (sh-bits), 0 };
  } else {
    return {
      U(x.lo) >> sh | x.hi << (bits - sh),
      x.hi >> sh
    };
  }
}

template <typename T>
__host__ __device__ __forceinline__ lohi<T> &operator<<=(lohi<T> &x, unsigned sh) {
  x = x << sh;
  return x;
}

template <typename T>
__host__ __device__ __forceinline__ lohi<T> &operator>>=(lohi<T> &x, unsigned sh) {
  x = x >> sh;
  return x;
}

template <typename T>
__host__ __device__ __forceinline__ lohi<T> operator-(lohi<T> a, lohi<T> b) {
  lohi<T> ret;
  ret.lo = a.lo - b.lo;
  int borrow = (b.lo && ret.lo > a.lo);
  ret.hi = a.hi - b.hi - borrow;
  return ret;
}

__host__ __device__ inline uint32_t div_lohi(uint32_t lo, uint32_t hi, uint32_t divisor) {
  return (static_cast<uint64_t>(hi) << 32 | lo) / divisor;
}

__host__ __device__ __forceinline__ lohi<uint64_t> mull(uint64_t a, uint64_t b) {
  lohi<uint64_t> ret;
#ifdef __CUDA_ARCH__
  ret.lo = a * b;
  ret.hi = __umul64hi(a, b);
#else
  unsigned __int128 m = (unsigned __int128)a * b;
  ret.lo = m;
  ret.hi = m >> 64;
#endif
  return ret;
}

template <typename uint>
__host__ __device__ int ilog2(uint x) noexcept {
  int n = 0;
  while (x >>= 1)
    n++;
  return n;
};

__host__ __device__ inline uint64_t div_lohi(uint64_t lo, uint64_t hi, uint64_t divisor) {
#if defined(__x86_64) && defined(__GNUC__) && !defined(__CUDA_ARCH__)
  // I hope this gets compiled to dividing rdx:rax register pair by a 64-bit value
  return (static_cast<unsigned __int128>(hi) << 64 | lo) / divisor;
#else
  // NVCC doesn't support __int128 in device code, so we need a bit of hackery
  if (!hi)  // No high part? Just divide in 64-bits and be done.
    return lo / divisor;

  // long division:

  int lnum = ilog2(hi) + 64;
  int lden = ilog2(divisor);

  lohi<uint64_t> num = { lo, hi };
  lohi<uint64_t> den = { divisor, 0 };

  // calculate MSB positions...
  int sh = lnum - lden;
  // .. and align numerator and denominator
  den <<= sh;

  uint64_t q = 0;

  while (sh >= 0) {
    lohi<uint64_t> dif = num - den;  // this serves both as difference and comparison
    if (static_cast<int64_t>(dif.hi) >= 0) {
      num = dif;
      q |= static_cast<uint64_t>(1) << sh;
    }
    sh--;
    den >>= 1;
  }
  return q;
#endif
}

}  // namespace detail

/**
 * @brief Fast unsigned integer division
 *
 * This class is useful when there's a need calculate division (and/or remainder) of many numbers
 * by common divisor, not known at compile time.
 *
 * Based on:
 * Labor of Division (Episode III): Faster Unsigned Division by Constants
 * ridiculous_fish corydoras@ridiculousfish.com
 * https://ridiculousfish.com/blog/posts/labor-of-division-episode-iii.html
 *
 * Limitations:
 * - doesn't work for maximum value of uint when dividing by "uncooperative" numbers
 */
template <typename uint>
struct fast_div {
  uint divisor;
  uint mul;
  uint8_t add;
  uint8_t shift;

  static_assert(std::is_integral<uint>::value && std::is_unsigned<uint>::value,
                "fast_div works only with unsigned integers");

  __host__ __device__ fast_div() {}

  __host__ __device__ fast_div(uint divisor) {  // NOLINT
    init(divisor);
  }

  __host__ __device__ void init(uint divisor) {
    this->divisor = divisor;
    this->mul = 1;
    this->shift = 0;
    this->add = 0;
    if (divisor == 0) {
      return;
    }

    int log_div = detail::ilog2(divisor);
    this->shift = log_div;

    if ((divisor & (divisor - 1)) == 0) {
      this->mul = 0;
      return;
    }

    uint m_lo = detail::div_lohi(0,                  uint(1) << log_div, divisor);
    uint m_hi = detail::div_lohi(uint(1) << log_div, uint(1) << log_div, divisor);
    this->add = (m_lo == m_hi) ? 1 : 0;  // round-up failed, use round-down method
    this->mul = m_hi;
  }

  __host__ __device__ __forceinline__ operator uint() const noexcept {
    return divisor;
  }
};

__host__ __device__ __forceinline__ uint32_t operator/(uint32_t x, fast_div<uint32_t> y) {
  // If the divisor is a power of 2, the multiplier would be 2^32, which is out of range
  // - therefore, powers of 2 get special treatment and the multiplication is skipped.
#ifdef __CUDA_ARCH__
  if (y.mul)
    x = __umulhi(x + y.add, y.mul);
  return x >> y.shift;
#else
  if (y.mul) {
    uint32_t hi = static_cast<uint64_t>(x + y.add) * y.mul >> 32;
    return hi >> y.shift;
  } else {
    return x >> y.shift;
  }
#endif
}

__host__ __device__ __forceinline__ uint64_t operator/(uint64_t x, fast_div<uint64_t> y) {
  // If the divisor is a power of 2, the multiplier would be 2^64, which is out of range
  // - therefore, powers of 2 get special treatment and the multiplication is skipped.
#ifdef __CUDA_ARCH__
  if (y.mul)
    x = __umul64hi(x + y.add, y.mul);
  return x >> y.shift;
#else
  if (y.mul) {
    uint64_t hi = static_cast<unsigned __int128>(x + y.add) * y.mul >> 64;
    return hi >> y.shift;
  } else {
    return x >> y.shift;
  }
#endif
}

template <typename uint>
__host__ __device__ __forceinline__ uint operator%(uint x, fast_div<uint> y) {
  return x - x / y * y;
}

template <typename uint>
__host__ __device__ __forceinline__ uint div_mod(uint &mod, uint dividend, fast_div<uint> divisor) {
  uint q = dividend / divisor;
  mod = dividend - q * divisor;
  return q;
}

}  // namespace dali

#endif  // DALI_CORE_FAST_DIV_H_
