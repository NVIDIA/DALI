// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_CORE_MM_DETAIL_UTIL_H_
#define DALI_CORE_MM_DETAIL_UTIL_H_

#include <cstddef>
#include <type_traits>
#include "dali/core/host_dev.h"

namespace dali {
namespace mm {
namespace detail {

/**
 * @brief An integral constant with a bit pattern used for detecting memory corruption.
 */
template <typename T, size_t sz = sizeof(T)>
struct sentinel_value;

template <typename T>
struct sentinel_value<T, 1> : std::integral_constant<T, static_cast<T>(0xAB)> {};

template <typename T>
struct sentinel_value<T, 2> : std::integral_constant<T, static_cast<T>(0xABCD)> {};

template <typename T>
struct sentinel_value<T, 4> : std::integral_constant<T, static_cast<T>(0xABCDABCDu)> {};

template <typename T>
struct sentinel_value<T, 8> : std::integral_constant<T, static_cast<T>(0xABCDABCDABCDABCDuL)> {};

/**
 * @brief Stores a sentinel value of type T at a given offset from a base pointer mem.
 */
template <typename T>
DALI_HOST_DEV
void write_sentinel(void *mem, ptrdiff_t offset = 0) {
  mem = static_cast<char*>(mem) + offset;
  *static_cast<T*>(mem) = sentinel_value<T>::value;
}

/**
 * @brief Checks whether there's a correct sentinel value of type T at a given offset
 *        from a base pointer mem.
 */
template <typename T>
DALI_HOST_DEV
bool check_sentinel(const void *mem, ptrdiff_t offset = 0) {
  mem = static_cast<const char*>(mem) + offset;
  return *static_cast<const T*>(mem) == sentinel_value<T>::value;
}

struct dummy_lock {
  constexpr void lock() const noexcept {}
  constexpr void unlock() const noexcept {}
  constexpr bool try_lock() const noexcept { return true; }
};

}  // namespace detail
}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_DETAIL_UTIL_H_
