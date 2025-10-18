// Copyright (c) 2018-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_UTIL_H_
#define DALI_CORE_UTIL_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <initializer_list>

#include "dali/core/int_literals.h"
#include "dali/core/host_dev.h"
#include "dali/core/force_inline.h"
#include "dali/core/span.h"

namespace dali {

using std::size_t;

template <typename T, size_t N>
DALI_HOST_DEV constexpr T *begin(T (&array)[N]) noexcept { return array; }
template <typename T, size_t N>
DALI_HOST_DEV constexpr T *end(T (&array)[N]) noexcept { return array + N; }

DALI_NO_EXEC_CHECK
template <typename T, typename = std::enable_if_t<!std::is_const<T>::value>>
DALI_HOST_DEV constexpr auto begin(T &collection)->decltype(collection.begin()) {
  return collection.begin();
}

DALI_NO_EXEC_CHECK
template <typename T, typename = std::enable_if_t<!std::is_const<T>::value>>
DALI_HOST_DEV constexpr auto end(T &collection)->decltype(collection.end()) {
  return collection.end();
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV constexpr auto begin(const T &collection)->decltype(collection.begin()) {
  return collection.begin();
}

DALI_NO_EXEC_CHECK
template <typename T>
DALI_HOST_DEV constexpr auto end(const T &collection)->decltype(collection.end()) {
  return collection.end();
}

DALI_NO_EXEC_CHECK
template <typename Target, typename Source>
DALI_HOST_DEV
void append(Target &target, const Source &source) {
  target.insert(dali::end(target), dali::begin(source), dali::end(source));
}

DALI_NO_EXEC_CHECK
template <typename Collection>
DALI_HOST_DEV auto size(const Collection &c)->decltype(c.size()) {
  return c.size();
}

template <typename T, size_t N>
DALI_HOST_DEV
size_t size(const T (&a)[N]) {
  return N;
}

template <typename Value, typename Alignment>
DALI_HOST_DEV
constexpr Value align_up(Value v, Alignment a) {
  return v + ((a - 1) & -v);
}

template <typename Value, typename Alignment>
DALI_HOST_DEV
constexpr Value alignment_offset(Value v, Alignment a) {
  return v & (a - 1);
}

template <typename Value, typename Alignment>
DALI_HOST_DEV
constexpr Value align_down(Value v, Alignment a) {
  return v & -a;
}

DALI_HOST_DEV
constexpr int32_t div_ceil(int32_t total, uint32_t grain) {
  return (total + grain - 1) / grain;
}

DALI_HOST_DEV
constexpr uint32_t div_ceil(uint32_t total, uint32_t grain) {
  return (total + grain - 1) / grain;
}

DALI_HOST_DEV
constexpr int64_t div_ceil(int64_t total, uint64_t grain) {
  return (total + grain - 1) / grain;
}

DALI_HOST_DEV
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

template <typename T>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr std::enable_if_t<std::is_integral<T>::value, T> next_pow2(T n) {
  T pow2 = 1;
  while (n > pow2) {
    pow2 += pow2;
  }
  return pow2;
}

template <typename T>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr std::enable_if_t<std::is_integral<T>::value, T> prev_pow2(T n) {
  T pow2 = 1;
  while (n - pow2 > pow2) {  // avoids overflow
    pow2 += pow2;
  }
  return pow2;
}

template <typename T>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr std::enable_if_t<std::is_integral<T>::value, bool> is_pow2(T n) {
  return (n & (n-1)) == 0;
}


/**
 * @brief Calculates the position of most significant bit in x
 * @return The position of MSB or 0 if x is 0.
 */
template <typename T>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr std::enable_if_t<std::is_integral<T>::value, int> ilog2(T x) {
  int n = 0;
  while (x >>= 1)
    n++;
  return n;
}


/**
 * @brief Count Trailing Zeros (CTZ)
 *
 * @return The number of trailing (LSB) zeros or the number of bits in the argument,
 *         if it's equal to 0.
 */
template <typename Integer>
DALI_HOST_DEV DALI_FORCEINLINE
std::enable_if_t<std::is_integral<Integer>::value, int> ctz(Integer word) {
  std::make_unsigned_t<Integer> uword = word;
  if (uword == 0)
    return sizeof(uword) * 8;
  // The hardware can check for multiple bits at a time, so we don't have to scan sequentially
  int bit = 0;
  if (sizeof(uword) > 4 && (uword & 0xffffffffu) == 0) {
    bit += 32;
    const int shift32 = sizeof(uword) > 4 ? 32 : 0;  // workaround undefined shift warning
    uword >>= shift32;
  }
  if (sizeof(uword) > 2 && (uword & 0xffffu) == 0) {
    bit += 16;
    const int shift16 = sizeof(uword) > 2 ? 16 : 0;  // workaround undefined shift warning
    uword >>= shift16;
  }
  if (sizeof(uword) > 1 && (uword & 0xffu) == 0) {
    bit += 8;
    const int shift8 = sizeof(uword) > 1 ? 8 : 0;  // workaround undefined shift warning
    uword >>= shift8;
  }
  if ((uword & 0xfu) == 0) {
    bit += 4;
    uword >>= 4;
  }
  if ((uword & 0x3u) == 0) {
    bit += 2;
    uword >>= 2;
  }
  if ((uword & 0x1u) == 0) {
    bit += 1;
    uword >>= 1;
  }
  return bit;
}


/**
 * @brief Returns an integer where bits at indicies in `bit_indices` are set to 1.
 * @remarks Indices that are outside the bit-width of OutType are ignored.
 */
DALI_NO_EXEC_CHECK
template <typename OutType = uint64_t, typename BitIndices>
DALI_HOST_DEV DALI_FORCEINLINE
OutType to_bit_mask(const BitIndices &bit_indices) {
  static_assert(std::is_integral<OutType>::value, "A bit mask must be of integral type");
  OutType mask = 0;
  for (int idx : bit_indices)
    mask |= OutType(1) << idx;
  return mask;
}

template <typename DependentName, typename Result>
using if_istype = std::conditional_t<false, DependentName, Result>;

template <typename Collection, typename T = void>
using if_iterable = if_istype<decltype(*dali::end(std::declval<Collection>())),
                              if_istype<decltype(*dali::begin(std::declval<Collection>())), T>>;

template <typename Collection, typename T = void>
using if_indexable = if_istype<decltype(std::declval<Collection>()[0]), T>;

template <typename Collection, typename T = void>
using if_array_like = if_indexable<Collection,
                                  if_istype<decltype(dali::size(std::declval<Collection>())), T>>;

template <typename It>
using is_integer_iterator = std::is_integral<
  std::remove_reference_t<decltype(*std::declval<It>())>>;

template <typename C>
using is_integer_collection = is_integer_iterator<
  decltype(dali::begin(std::declval<C>()))>;

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


// Imported from C++20 https://en.cppreference.com/w/cpp/utility/functional/identity
struct identity {
  template <typename T>
  DALI_HOST_DEV DALI_FORCEINLINE
  T &&operator()(T &&x) const noexcept {
    return std::forward<T>(x);
  }
};

/** Implements a trait that checks for the presence of a member called <member_name>
 *
 * Usage:
 * ```
 * // at namespace/global scope
 * IMPL_HAS_MEMBER(foo);
 *
 * struct S {
 *     int foo;
 * };
 *
 * template <typename X>
 * auto foo_or_zero(X x) {
 *   if constexpr (has_member_foo_v<X>)
 *     return x.foo;
 *   else
 *     return 0;  // no foo in x
 * }
 *
 * int main() {
 *   S s { 42 };
 *   cout << foo_or_zero(s) << endl;  // 42
 *   cout << foo_or_zero(1.234) << endl;  // 0
 * }
 *
 * ```
 */
#define IMPL_HAS_MEMBER(member_name)\
template <typename T, typename = decltype(std::declval<T>().member_name)>\
std::true_type HasMember_##member_name(T *);\
std::false_type HasMember_##member_name(...);\
template <typename T>\
using has_member_##member_name = \
  decltype(HasMember_##member_name(std::declval<std::add_pointer_t<T>>()));\
template <typename T>\
constexpr bool has_member_##member_name##_v = has_member_##member_name<T>::value

#define IMPL_HAS_NESTED_TYPE(type_name)\
template <typename T>\
std::true_type HasNested_##type_name(typename T::type_name *);\
template <typename T>\
std::false_type HasNested_##type_name(...);\
template <typename T>\
struct has_type_##type_name : decltype(HasNested_##type_name<T>(nullptr)) {}; \

#define IMPL_HAS_UNIQUE_STATIC_FUNCTION(function_name)\
template <typename T>\
std::is_function<decltype(T::function_name)> HasUniqueStaticFunction_##function_name(T *);\
template <typename T>\
std::false_type HasUniqueStaticFunction_##function_name(...);\
template <typename T>\
struct has_unique_static_function_##function_name : \
  decltype(HasUniqueStaticFunction_##function_name<T>(nullptr)) {}; \

#define IMPL_HAS_UNIQUE_MEMBER_FUNCTION(function_name)\
template <typename T>\
std::is_member_function_pointer<decltype(&T::function_name)>\
HasUniqueMemberFunction_##function_name(T *);\
template <typename T>\
std::false_type HasUniqueMemberFunction_##function_name(...);\
template <typename T>\
struct has_unique_member_function_##function_name : \
  decltype(HasUniqueMemberFunction_##function_name<T>(nullptr)) {}; \

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
  std::remove_const_t<std::remove_reference_t<ExtentType>>>::type;

/// @brief Returns the product of all elements in shape
/// @param shape_begin - start of the shape extent list
/// @param shape_end - end of the shape extent list
DALI_NO_EXEC_CHECK
template <typename Iter>
DALI_HOST_DEV
inline volume_t<decltype(*std::declval<Iter>())>
volume(Iter shape_begin, Iter shape_end) {
  if (shape_begin == shape_end)
    return 1;
  auto it = shape_begin;
  volume_t<decltype(*shape_begin)> v = *it;
  for (++it; it != shape_end; ++it)
    v *= *it;
  return v;
}

/// @brief Returns the product of all elements in shape
/// @param shape - an iterable collection of extents
DALI_NO_EXEC_CHECK
template <typename Shape>
DALI_HOST_DEV
inline auto volume(const Shape &shape)->decltype(volume(dali::begin(shape), dali::end(shape))) {
  return volume(dali::begin(shape), dali::end(shape));
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
DALI_HOST_DEV
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


template <size_t N, typename T>
std::array<T, N> uniform_array(const T& t) {
  std::array<T, N> result;
  result.fill(t);
  return result;
}

// flattening of spans of statically-sized arrays

template <typename T, span_extent_t extent>
inline span<T, extent> flatten(span<T, extent> in) {
  return in;
}

template <size_t N, typename T, span_extent_t extent>
inline auto flatten(span<std::array<T, N>, extent> in) {
  const span_extent_t next_extent = extent == dynamic_extent
    ? dynamic_extent : extent*span_extent_t(N);
  return flatten(span<T, next_extent>(&in[0][0], in.size() * N));
}

template <size_t N, typename T, span_extent_t extent>
inline auto flatten(span<const std::array<T, N>, extent> in) {
  const span_extent_t next_extent = extent == dynamic_extent
    ? dynamic_extent : extent*span_extent_t(N);
  return flatten(span<const T, next_extent>(&in[0][0], in.size() * N));
}

template <typename T>
using is_pod = std::conjunction<
    std::is_standard_layout<T>,
    std::is_trivially_copy_assignable<T>,
    std::is_trivially_copyable<T>,
    std::is_trivially_default_constructible<T>,
    std::is_trivially_destructible<T>>;

template <typename T>
constexpr bool is_pod_v = is_pod<T>::value;

}  // namespace dali

#endif  // DALI_CORE_UTIL_H_
