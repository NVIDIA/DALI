// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
#include <initializer_list>

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


constexpr int32_t div_ceil(int32_t total, uint32_t grain) {
  return (total + grain - 1) / grain;
}

constexpr uint32_t div_ceil(uint32_t total, uint32_t grain) {
  return (total + grain - 1) / grain;
}

constexpr int64_t div_ceil(int64_t total, uint64_t grain) {
  return (total + grain - 1) / grain;
}

constexpr uint64_t div_ceil(uint64_t total, uint64_t grain) {
  return (total + grain - 1) / grain;
}

static_assert(div_ceil(0, 32) == 0, "Should not change");
static_assert(div_ceil(1, 32) == 1, "Should align up");
static_assert(div_ceil(32, 32) == 1, "Should not align up");
static_assert(div_ceil(65, 64) == 2, "Should align up");

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


/// @brief Type of a volume with given `ExtentType`
template <typename ExtentType, bool arithm = std::is_arithmetic<ExtentType>::value>
struct volume_type {
  using type = decltype(std::declval<ExtentType>() * std::declval<ExtentType>());
};

/// @brief volume_type is undefined for non-arithmetic types, by default
template <typename ExtentType>
struct volume_type<ExtentType, false> {};

/// @brief 32-bit integers need promotion to 64-bit to store volume safely
template <>
struct volume_type<int32_t, true> {
  using type = int64_t;
};

/// @brief 32-bit integers need promotion to 64-bit to store volume safely
template <>
struct volume_type<uint32_t, true> {
  using type = uint64_t;
};

template <typename ExtentType>
using volume_t = typename volume_type<
  typename std::remove_const<typename std::remove_reference<ExtentType>::type>::type>::type;

/// @brief Returns the product of all elements in shape
/// @param shape_begin - start of the shape extent list
/// @param shape_end - end of the shape extent list
template <typename Iter>
inline volume_t<decltype(*std::declval<Iter>())>
volume(Iter shape_begin, Iter shape_end) {
  if (shape_begin == shape_end)
    return 0;  // perhaps we should return 1 as a neutral element of multiplication?
  auto it = shape_begin;
  volume_t<decltype(*shape_begin)> v = *it;
  for (++it; it != shape_end; ++it)
    v *= *it;
  return v;
}

/// @brief Returns the product of all elements in shape
/// @param shape - an iterable collection of extents
template <typename Shape>
inline auto volume(const Shape &shape)->decltype(volume(std::begin(shape), std::end(shape))) {
  return volume(std::begin(shape), std::end(shape));
}

/// @brief Returns the product of all elements in shape
/// @param shape - an initializer_list of extents
template <typename Extent>
inline volume_t<Extent> volume(std::initializer_list<Extent> shape) {
  return volume(shape.begin(), shape.end());
}

/// @brief Returns the argument, promoted to an appropriate volume_t
/// @param single_dim - the sole dimension to be returned as a volume
template <typename Extent>
constexpr volume_t<Extent> volume(Extent single_dim) {
  return single_dim;
}

/// @brief Test if any of the values in the variadic template boolean values list is true
template <bool... values>
struct any_of : std::false_type {};
template <bool... tail>
struct any_of<true, tail...> : std::true_type {};
template <bool... tail>
struct any_of<false, tail...> : any_of<tail...> {};

static_assert(any_of<false, true, false>::value,
              "Should return true_type when one of the values is true.");
static_assert(!any_of<false, false, false>::value,
              "Should return false_type when all the values are false.");


/// @brief Test if all the values in the variadic template boolean values list are true
template <bool... values>
struct all_of : std::true_type {};
template <bool... tail>
struct all_of<false, tail...> : std::false_type {};
template <bool... tail>
struct all_of<true, tail...> : all_of<tail...> {};

static_assert(all_of<true, true, true>::value,
              "Should return true_type when all the values are true.");
static_assert(!all_of<true, false, true>::value,
              "Should return false_type when any of the values is false.");

}  // namespace dali

#endif  // DALI_KERNELS_UTIL_H_
