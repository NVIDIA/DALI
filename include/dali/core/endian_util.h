// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_ENDIAN_UTIL_H_
#define DALI_CORE_ENDIAN_UTIL_H_

#ifdef __linux__
#include <endian.h>
#endif
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <type_traits>
#include <cstring>
#include <array>
#include <tuple>
#include <utility>

namespace dali {

template <typename T>
struct swap_endian_impl {
  static void do_swap(T &t) {
    static_assert(std::is_fundamental<T>::value || std::is_enum<T>::value);
    if (sizeof(t) > 1) {
      char src[sizeof(T)];
      char dst[sizeof(T)];
      std::memcpy(src, &t, sizeof(t));
      for (size_t i = 0; i < sizeof(t); i++)
          dst[i] = src[sizeof(t) - 1 - i];
      std::memcpy(&t, dst, sizeof(t));
    }
  }
};

/**
 * @brief Swaps endianness of an object
 *
 * This CPO (customization point object) is a function object that swaps the endianness
 * of the argument.
 *
 * Usage:
 * ```
 * int x = 0x11223344;
 * swap_endian(x);
 * assert(x == 0x44332211);
 * ```
 */
constexpr struct swap_endian_cpo {
  template <typename T>
  void operator()(T &t) const {
    swap_endian_impl<T>::do_swap(t);
  }
} swap_endian;

template <typename T, size_t N>
struct swap_endian_impl<T[N]> {
  static void do_swap(T a[N]) {
    for (size_t i = 0; i < N; i++)
      swap_endian(a[i]);
  }
};

template <typename T, size_t N>
struct swap_endian_impl<std::array<T, N>> {
  static void do_swap(std::array<T, N> &a) {
    for (size_t i = 0; i < N; i++)
      swap_endian(a[i]);
  }
};

template <typename T, typename U>
struct swap_endian_impl<std::pair<T, U>> {
  static void do_swap(std::pair<T, U> &p) {
    swap_endian(p.first);
    swap_endian(p.second);
  }
};

template <typename... T>
struct swap_endian_impl<std::tuple<T...>> {
  static void do_swap(std::tuple<T...> &t) {
    do_swap(t, std::integral_constant<size_t, 0>());
  }
  template <size_t idx>
  static void do_swap(std::tuple<T...> &t, std::integral_constant<size_t, idx>) {
    swap_endian(std::get<idx>(t));
    do_swap(t, std::integral_constant<size_t, idx+1>());
  }
  static void do_swap(std::tuple<T...> &t, std::integral_constant<size_t, sizeof...(T)>) {
  }
};

#ifdef __linux__
#if __BYTE_ORDER == __LITTLE_ENDIAN
/// @brief The host machine is little-endian
static constexpr bool is_little_endian = true;
/// @brief The host machine is big endian
static constexpr bool is_big_endian = false;
#elif __BYTE_ORDER == __BIG_ENDIAN
/// @brief The host machine is little-endian
static constexpr bool is_little_endian = false;
/// @brief The host machine is big endian
static constexpr bool is_big_endian = true;
#else
#error "Cannot establish endianness of the target system."
#endif
#else
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmultichar"
#endif
#if ('ABCD' == 0x41424344UL)
/// @brief The host machine is little-endian
static constexpr bool is_little_endian = true;
/// @brief The host machine is big endian
static constexpr bool is_big_endian = false;
#elif ('ABCD' == 0x44434241UL)
/// @brief The host machine is little-endian
static constexpr bool is_little_endian = false;
/// @brief The host machine is big endian
static constexpr bool is_big_endian = true;
#else
#error "Cannot establish endianness of the target system."
#endif
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#endif

#define DALI_SWAP_FIELD_ENDIANNESS(unused, unused2, field) swap_endian(s.field);

/**
 * @brief Used for generating field-wise endianness swapping function for a structure _Struct
 *
 * Usage:
 * ```
 * struct Size {
 *   int16_t width, heigth;
 * };
 * SWAP_ENDIAN_FIELDS(Size, width, heigth);
 *
 * Size s = { 640, 480 };
 * to_big_endian(s);                // convert to big-endian
 * fwrite(&s, sizeof(s), 1, file);  // store in a big-endian file
 * ```
 *
 * NOTE: It does not support inheritance - the fields from the base class have to
 *       be enumerated as well
 */
#define SWAP_ENDIAN_FIELDS(_Struct, ...)\
template <>\
void swap_endian_impl<_Struct>::do_swap(_Struct &s) {\
  BOOST_PP_SEQ_FOR_EACH(DALI_SWAP_FIELD_ENDIANNESS, unused, \
    BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))\
}\

/**
 * @brief Converts (in place) the object to little endian
 *
 * Converts a structure that is stored in host order to little-endian order,
 * e.g. for writing it to a file.
 * If the host system is little endian, this function is a no-op.
 */
template <typename T>
void to_little_endian(T &t) {
  if (is_big_endian)
    swap_endian(t);
}

/**
 * @brief Converts (in place) the object from little endian
 *
 * Converts a structure that is stored in little-endian order, e.g. read
 * from a file, to host order.
 * If the host system is little endian, this function is a no-op.
 */
template <typename T>
void from_little_endian(T &t) {
  if (is_big_endian)
    swap_endian(t);
}

/**
 * @brief Converts (in place) the object to big endian
 *
 * Converts a structure that is stored in host order to big-endian order,
 * e.g. for writing it to a file.
 * If the host system is big endian, this function is a no-op.
 */
template <typename T>
void to_big_endian(T &t) {
  if (is_little_endian)
    swap_endian(t);
}

/**
 * @brief Converts (in place) the object from big endian
 *
 * Converts a structure that is stored in big-endian order, e.g. read
 * from a file, to host order.
 * If the host system is big endian, this function is a no-op.
 */
template <typename T>
void from_big_endian(T &t) {
  if (is_little_endian)
    swap_endian(t);
}

}  // namespace dali

#endif  // DALI_CORE_ENDIAN_UTIL_H_
