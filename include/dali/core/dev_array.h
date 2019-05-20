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

#ifndef DALI_CORE_DEV_ARRAY_H_
#define DALI_CORE_DEV_ARRAY_H_

#include <cuda_runtime.h>
#include <array>

namespace dali {

template <typename T, size_t N>
class DeviceArray {
 public:
  constexpr DeviceArray() = default;
  __host__ DeviceArray(const std::array<T, N> &src) {
    for (size_t i = 0; i < N; i++)
      data_[i] = src[i];
  }

  __host__ operator std::array<T, N>() const noexcept {
    std::array<T, N> ret;
    for (size_t i = 0; i < N; i++)
      ret[i] = data_[i];
    return ret;
  }

  using value_type = T;
  using reference = T&;
  using const_reference = const T&;
  using iterator = T*;
  using const_iterator = const T*;

  __host__ __device__ T &operator[](ptrdiff_t index)
  { return data_[index]; }
  __host__ __device__ constexpr const T &operator[](ptrdiff_t index) const
  { return data_[index]; }

  __host__ __device__ inline T *begin() { return data_; }
  __host__ __device__ constexpr const T *begin() const { return data_; }
  __host__ __device__ constexpr const T *cbegin() const { return data_; }
  __host__ __device__ inline T *end() { return data_ + N; }
  __host__ __device__ constexpr const T *end() const { return data_ + N; }
  __host__ __device__ constexpr const T *cend() const { return data_ + N; }
  __host__ __device__ constexpr size_t size() const { return N; }
  __host__ __device__ constexpr bool empty() const { return N == 0; }
  __host__ __device__ inline T *data() { return data_; }
  __host__ __device__ constexpr const T *data() const { return data_; }

  __host__ __device__ inline bool operator==(const DeviceArray &other) const {
    for (size_t i = 0; i < N; i++) {
      if (data_[i] != other.data_[i])
        return false;
    }
    return true;
  }

  __host__ __device__ inline bool operator!=(const DeviceArray &other) const {
    return !(*this == other);
  }

 private:
  T data_[N];
};

template <typename T>
class DeviceArray<T, 0> {
 public:
  constexpr DeviceArray() = default;
  __host__ DeviceArray(const std::array<T, 0> &) {
  }

  constexpr __host__ operator std::array<T, 0>() const noexcept {
    return {};
  }

  using value_type = T;
  using reference = T&;
  using const_reference = const T&;
  using iterator = T*;
  using const_iterator = const T*;

  __host__ __device__ T &operator[](ptrdiff_t index)
  { return data()[index]; }
  __host__ __device__ constexpr const T &operator[](ptrdiff_t index) const
  { return data()[index]; }

  __host__ __device__ inline T *begin() { return data(); }
  __host__ __device__ constexpr const T *begin() const { return data(); }
  __host__ __device__ constexpr const T *cbegin() const { return data(); }
  __host__ __device__ inline T *end() { return data(); }
  __host__ __device__ constexpr const T *end() const { return data(); }
  __host__ __device__ constexpr const T *cend() const { return data(); }
  __host__ __device__ constexpr size_t size() const { return 0; }
  __host__ __device__ constexpr bool empty() const { return true; }
  __host__ __device__ inline T *data() { return reinterpret_cast<T*>(this); }
  __host__ __device__ constexpr const T *data() const { return reinterpret_cast<const T*>(this); }
};

}  // namespace dali

#endif  // DALI_CORE_DEV_ARRAY_H_
