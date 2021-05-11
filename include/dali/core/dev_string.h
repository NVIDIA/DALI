// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_CORE_DEV_STRING_H_
#define DALI_CORE_DEV_STRING_H_

#include <cuda_runtime.h>
#include <type_traits>
#include "dali/core/cuda_utils.h"

namespace dali {

struct DeviceString {
  __device__ DeviceString() {}
  __device__ DeviceString(const char *text) {
    while (text[length_])
      length_++;
    if (length_) {
      data_ = static_cast<char*>(malloc(length_+1));
      for (size_t i = 0; i <= length_; i++)
        data_[i] = text[i];
    }
  }
  __device__ void reset(char *data, size_t length) {
    clear();
    data_ = data;
    length_ = length;
  }

  __device__ DeviceString(const DeviceString &other) {
    *this = other;
  }
  __device__ DeviceString(DeviceString &&other) {
    *this = cuda_move(other);
  }

  __device__ ~DeviceString() {
    clear();
  }

  __device__ void clear() {
    if (length_) {
      free(data_);
      data_ = nullptr;
      length_ = 0;
    }
  }

  __device__ const char *c_str() const {
    return data_ ? data_ : "";
  }

  __device__ const char *data() const {
    return data_;
  }

  __device__ size_t size() const {
    return length_;
  }

  __device__ size_t length() const {
    return length_;
  }

  __device__ DeviceString &operator=(const DeviceString &other) {
    if (&other != this) {
      clear();
      length_ = other.length_;
      if (length_ > 0) {
        data_ = static_cast<char*>(malloc(length_+1));
        for (size_t i = 0; i <= length_; i++)
          data_[i] = other.data_[i];
      }
    }
    return *this;
  }

  __device__ DeviceString operator+(const DeviceString &other) {
    size_t l1 = length();
    size_t l2 = other.length();
    if (!l2)
      return *this;
    if (!l1)
      return other;
    DeviceString result;
    result.data_ = static_cast<char*>(malloc(l1+l2+1));
    for (size_t i = 0; i < l1; i++)
      result.data_[i] = data_[i];
    for (size_t i = 0; i < l2; i++)
      result.data_[i + l1] = other.data_[i];
    result.data_[l1+l2] = 0;
    result.length_ = l1+l2;
    return result;
  }

  __device__ DeviceString &operator=(DeviceString &&other) {
    cuda_swap(data_, other.data_);
    cuda_swap(length_, other.length_);
    return *this;
  }

  __device__ DeviceString &operator+=(const DeviceString &other) {
    if (!other.length())
      return *this;
    if (!length())
      return *this = other;
    return *this = (*this + other);
  }

  __device__ char &operator[](ptrdiff_t idx) { return data_[idx]; }
  __device__ const char &operator[](ptrdiff_t idx) const { return data_[idx]; }

  char *data_ = nullptr;
  size_t length_ = 0;
};

constexpr __device__ const char *dev_to_string(char *literal) { return literal; }
constexpr __device__ const char *dev_to_string(const char *literal) { return literal; }
inline __device__ DeviceString dev_to_string(bool b) { return b ? "true" : "false"; }

inline __device__ DeviceString dev_to_string(long long x) {  // NOLINT
  static_assert(sizeof(long long) == 8,  // NOLINT
                "This function does not work when long long is not 64-bit");
  if (x == 0)
    return "0";
  char buf[32];
  int cursor = 31;
  buf[31] = 0;
  bool neg = false;
  if (x < 0) {
    neg = true;
    x = -x;
    if (x < 0) {
      // number is self-negative - it must be -2^63
      return "-9223372036854775808";
    }
  }
  while (x) {
    int digit = x%10;
    x /= 10;
    buf[--cursor] = digit + '0';
  }
  if (neg)
    buf[--cursor] = '-';
  return buf+cursor;
}

template <typename T>
__device__ std::enable_if_t<std::is_integral<T>::value, DeviceString>
dev_to_string(T x) {
  return dev_to_string(static_cast<long long>(x));  // NOLINT
}


inline __device__ DeviceString dev_to_string(const void *ptr) {
  if (ptr == nullptr)
    return "0x0";
  uintptr_t x = reinterpret_cast<uintptr_t>(ptr);
  char buf[20];
  int cursor = 19;
  buf[19] = 0;
  while (x) {
    int digit = x&0xf;
    x >>= 4;;
    buf[--cursor] = "01234567889ABCDEF"[digit];
  }
  buf[--cursor] = 'x';
  buf[--cursor] = '0';
  return buf+cursor;
}

template <typename F>
inline __device__ std::enable_if_t<std::is_floating_point<F>::value, DeviceString>
dev_to_string(F x) {
  if (x == 0)
    return "0";
  char buf[64];;
  int mid = 32;
  int lcursor = mid;
  buf[mid] = 0;

  bool neg = false;
  if (x < 0) {
    neg = true;
    x = -x;
  }

  int exponent = 0;
  if (x > F(1e+8)) {
    F div = (1e+7);
    exponent = 7;
    while (x / div >= 10) {
      div *= 10;
      exponent++;
    }
    x /= div;
  } else if (x < 1e-4) {
    F mul = F(1e+4);
    exponent = -4;
    while (x * mul < 1) {
      mul *= 10;
      exponent--;
    }
    x *= mul;
  }

  int integer_part = x;
  F frac = x - integer_part;
  if (integer_part == 0) {
    buf[--lcursor] = '0';
  } else {
    while (integer_part) {
      int digit = integer_part%10;
      integer_part /= 10;
      buf[--lcursor] = digit + '0';
    }
  }

  int rcursor = mid;
  if (frac) {
    frac *= 10;
    buf[rcursor++] = '.';
    F thresh;
    if (std::is_same<F, float>::value) {
      thresh = x * (10.0f / (1<<23));
    } else {
      thresh = x * (10.0 / (1LL<<52));
    }
    int digit = 0;
    int max_prec = 7;
    for (int prec = 0; frac > thresh && prec < max_prec; prec++) {
      digit = frac;
      thresh *= 10;
      buf[rcursor++] = '0' + digit;
      frac = 10 * (frac -  digit);
    }
  }

  if (exponent) {
    buf[rcursor++] = 'e';
    buf[rcursor++] = exponent > 0 ? '+' : '-';
    int exp_digits = 0;
    exponent = abs(exponent);
    for (int tmp = exponent; tmp; tmp /= 10, exp_digits++) {}

    int ecursor = rcursor + exp_digits;
    for (int tmp = exponent; tmp; tmp /= 10, exp_digits++)
      buf[--ecursor] = '0' + tmp%10;
    rcursor += exp_digits;
  }

  buf[rcursor] = 0;
  if (neg)
    buf[--lcursor] = '-';
  return buf+lcursor;
}

}  // namespace dali

#endif  // DALI_CORE_DEV_STRING_H_
