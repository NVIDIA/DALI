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

#ifndef DALI_KERNELS_SPAN_H_
#define DALI_KERNELS_SPAN_H_

#include <cstddef>

namespace tensor {

template <typename T, size_t static_size = (size_t)-1>
struct span {
  span() = default;
  span(T *p, size_t n = static_size) : p(p) { assert(n == static_size); }

  const T *p = nullptr;

  constexpr T *base() const { return p; }
  constexpr T &operator[](ptrdiff_t index) const { return p[index]; }
  constexpr T *begin() const { return p; }
  constexpr T *end() const { return p + size(); }
  constexpr size_t size() const { return static_size; }
};

template <typename T>
struct span<T, (size_t)-1> {
  span() = default;
  span(const T *p, size_t n) : p(p), n(n) {}

  const T *p = nullptr;
  const size_t n = 0;

  constexpr T *base() const { return p; }
  constexpr T &operator[](ptrdiff_t index) const { return p[index]; }
  constexpr T *begin() const { return p; }
  constexpr T *end() const { return p + size(); }
  constexpr size_t size() const { return n; }
};

}  // namespace tensor

#endif  // DALI_KERNELS_SPAN_H_
