// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
namespace kernels {

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


template <typename DependentName, typename Result>
using if_istype = typename std::conditional<false, DependentName, Result>::type;

template <typename Collection, typename T>
using if_iterable = if_istype<decltype(*std::end(std::declval<Collection>())),
                              if_istype<decltype(*std::begin(std::declval<Collection>())), T>>;

template <typename C>
struct element_type {
  using type = typename C::value_type;
};

template <typename C>
using element_t = typename element_type<C>::type;

// collection element type

template <typename C>
struct element_type<C&> : element_type<C> {};
template <typename C>
struct element_type<C&&> : element_type<C> {};
template <typename C>
struct element_type<const C> : element_type<C> {};
template <typename C>
struct element_type<volatile C> : element_type<C> {};

template <typename T, size_t N>
struct element_type<T[N]> {
  using type = T;
};

template <typename T>
struct same_as { using type = T; };
/// @brief Used in template functions to make some parameters more important in type inference
///
/// Usage:
/// ```
/// template <typename T>
/// void Fill1(std::vector<T> &dest, const T &value);
/// template <typename T>
/// void Fill2(std::vector<T> &dest, same_as<const T &> value);
/// ...
/// std::vector<float> v;
/// Fill1(v, 0);  // error: ambiguous deduction of T (float vs int)
/// Fill2(v, 0);  // OK, value is float as inferred from v
/// ```
template <typename T>
using same_as_t = typename same_as<T>::type;

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

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_UTIL_H_
