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

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <type_traits>
#include <cstring>

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

#if ('ABCD' == 0x41424344UL)
static constexpr bool is_little_endian = true;
static constexpr bool is_big_endian = false;
#elif ('ABCD' == 0x44434241UL)
static constexpr bool is_little_endian = false;
static constexpr bool is_big_endian = true;
#else
#error "Cannot establish endianness of the target system."
#endif

template <typename T>
void swap_endian(T &t) {
  swap_endian_impl<T>::do_swap(t);
}

#define DALI_SWAP_FIELD_ENDIANNESS(unused, unused2, field) swap_endian(s.field);

#define SWAP_ENDIAN_FIELDS(_Struct, ...)\
template <>\
void swap_endian_impl<_Struct>::do_swap(_Struct &s) {\
  BOOST_PP_SEQ_FOR_EACH(DALI_SWAP_FIELD_ENDIANNESS, unused, \
    BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))\
}\

template <typename T>
void to_little_endian(T &t) {
  if (is_big_endian)
    swap_endian(t);
}

template <typename T>
void from_little_endian(T &t) {
  if (is_big_endian)
    swap_endian(t);
}

template <typename T>
void to_big_endian(T &t) {
  if (is_little_endian)
    swap_endian(t);
}

template <typename T>
void from_big_endian(T &t) {
  if (is_little_endian)
    swap_endian(t);
}

}  // namespace dali

#endif  // DALI_CORE_ENDIAN_UTIL_H_
