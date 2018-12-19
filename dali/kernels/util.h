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

#ifndef DALI_KERNELS_UTIL_H_
#define DALI_KERNELS_UTIL_H_

#include <cstddef>
#include <utility>

namespace dali {

using std::size_t;

template <typename Target, typename Source>
void append(Target &target, const Source &source) {
  target.insert(std::end(target), std::begin(source), std::end(source));
}

template <typename Collection>
auto size(const Collection &c)->decltype(c.size()) {
  return c.size();
}

template <typename T, size_t N>
size_t size(const T (&a)[N]) {
  return N;
}

template <typename Value, typename Alignment>
constexpr Value align_up(Value v, Alignment a) {
  return v + ((a - 1) & -v);
}

static_assert(align_up(17, 16) == 32, "Should align up");
static_assert(align_up(8, 8) == 8, "Should be already aligned");
static_assert(align_up(5, 8) == 8, "Should align");

constexpr ptrdiff_t dynamic_extent = -1;

template <typename T, ptrdiff_t static_extent = dynamic_extent>
struct span {
  span() = default;
  span(T *p, ptrdiff_t n = static_extent) : p(p) { assert(n == static_extent); }  // NOLINT

  T * const p = nullptr;

  constexpr T *data() const { return p; }
  constexpr T &operator[](ptrdiff_t index) const { return p[index]; }
  constexpr T *begin() const { return p; }
  constexpr T *end() const { return p + size(); }
  constexpr ptrdiff_t size() const { return static_extent; }
};

template <typename T>
struct span<T, dynamic_extent> {
  span() = default;
  span(T *p, ptrdiff_t n) : p(p), n(n) {}

  T *const p = nullptr;
  const ptrdiff_t n = 0;

  constexpr T *data() const { return p; }
  constexpr T &operator[](ptrdiff_t index) const { return p[index]; }
  constexpr T *begin() const { return p; }
  constexpr T *end() const { return p + size(); }
  constexpr ptrdiff_t size() const { return n; }
};


#define IMPL_HAS_NESTED_TYPE(type_name)\
template <typename T>\
std::true_type HasNested_##type_name(typename T::type_name *);\
template <typename T>\
std::false_type HasNested_##type_name(...);\
template <typename T>\
struct has_type_##type_name : decltype(HasNested_##type_name<T>(nullptr)) {}; \

#define IMPL_HAS_UNIQUE_FUNCTION(function_name)\
template <typename T>\
std::is_function<decltype(T::function_name)> HasUniqueFunction_##function_name(T *);\
template <typename T>\
std::false_type HasUniqueFunction_##function_name(...);\
template <typename T>\
struct has_unique_function_##function_name : \
  decltype(HasUniqueFunction_##function_name<T>(nullptr)) {}; \


}  // namespace dali

#endif  // DALI_KERNELS_UTIL_H_
