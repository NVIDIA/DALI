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
#include "dali/core/util.h"

namespace dali {

template <typename T, size_t N>
class DeviceArray {
 public:
  constexpr DeviceArray() = default;
  __host__ DeviceArray(const std::array<T, N> &src)
  noexcept(noexcept(std::declval<T*>()[0] = src[0])) {
    for (size_t i = 0; i < N; i++)
      data_[i] = src[i];
  }

  template <typename... Args>
  DALI_HOST_DEV DeviceArray(const T &arg0, const Args&... args)
  : data_{arg0, args...} {
    static_assert(sizeof...(Args) == N-1, "Wrong number of initializers");
  }

  __host__ operator std::array<T, N>() const
  noexcept(noexcept(std::array<T, N>()[0] = *static_cast<T*>(nullptr))) {
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
  using pointer = value_type *;
  using const_pointer = const value_type *;

  DALI_HOST_DEV T &operator[](ptrdiff_t index) noexcept
  { return data_[index]; }
  DALI_HOST_DEV constexpr const T &operator[](ptrdiff_t index) const noexcept
  { return data_[index]; }

  DALI_HOST_DEV inline iterator begin() noexcept { return data_; }
  DALI_HOST_DEV constexpr const_iterator begin() const noexcept { return data_; }
  DALI_HOST_DEV constexpr const_iterator cbegin() const noexcept { return data_; }
  DALI_HOST_DEV inline iterator end() noexcept { return data_ + N; }
  DALI_HOST_DEV constexpr const_iterator end() const noexcept { return data_ + N; }
  DALI_HOST_DEV constexpr const_iterator cend() const noexcept { return data_ + N; }
  DALI_HOST_DEV constexpr size_t size() const noexcept { return N; }
  DALI_HOST_DEV constexpr bool empty() const noexcept { return N == 0; }
  DALI_HOST_DEV inline pointer data() noexcept { return data_; }
  DALI_HOST_DEV constexpr const_pointer data() const noexcept { return data_; }
  DALI_HOST_DEV inline reference front() noexcept { return *data_; }
  DALI_HOST_DEV constexpr const_reference front() const noexcept { return *data_; }
  DALI_HOST_DEV inline reference back() noexcept { return data_[N-1]; }
  DALI_HOST_DEV constexpr const_reference back() const noexcept { return data_[N-1]; }

  DALI_HOST_DEV inline bool operator==(const DeviceArray &other) const noexcept {
    for (size_t i = 0; i < N; i++) {
      if (data_[i] != other.data_[i])
        return false;
    }
    return true;
  }

  DALI_HOST_DEV inline bool operator!=(const DeviceArray &other) const noexcept {
    return !(*this == other);
  }

 private:
  T data_[N];
};

template <typename T>
class DeviceArray<T, 0> {
 public:
  constexpr DeviceArray() = default;

  DALI_HOST_DEV DeviceArray(const std::array<T, 0> &) noexcept {}

  constexpr DALI_HOST_DEV operator std::array<T, 0>() const noexcept { return {}; }

  using value_type = T;
  using reference = T&;
  using const_reference = const T&;
  using iterator = T*;
  using const_iterator = const T*;

  DALI_HOST_DEV T &operator[](ptrdiff_t index) noexcept
  { return data()[index]; }
  DALI_HOST_DEV constexpr const T &operator[](ptrdiff_t index) const noexcept
  { return data()[index]; }

  DALI_HOST_DEV inline T *begin() noexcept { return data(); }
  DALI_HOST_DEV constexpr const T *begin() const noexcept { return data(); }
  DALI_HOST_DEV constexpr const T *cbegin() const noexcept { return data(); }
  DALI_HOST_DEV inline T *end() noexcept { return data(); }
  DALI_HOST_DEV constexpr const T *end() const noexcept { return data(); }
  DALI_HOST_DEV constexpr const T *cend() const noexcept { return data(); }
  DALI_HOST_DEV constexpr size_t size() const noexcept { return 0; }
  DALI_HOST_DEV constexpr bool empty() const noexcept { return true; }

  DALI_HOST_DEV inline T *data() noexcept {
    return reinterpret_cast<T*>(this);
  }
  DALI_HOST_DEV constexpr const T *data() const noexcept {
    return reinterpret_cast<const T*>(this);
  }
};


template <typename T>
constexpr DALI_HOST_DEV volume_t<T> volume(const DeviceArray<T, 0> &) {
  return 1;
}

template <typename T, size_t N>
DALI_HOST_DEV volume_t<T> volume(const DeviceArray<T, N> &arr) {
  volume_t<T> v = arr[0];
  for (size_t i = 1; i < N; i++) {
    v *= arr[i];
  }
  return v;
}

}  // namespace dali

#endif  // DALI_CORE_DEV_ARRAY_H_
