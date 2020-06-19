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

#ifndef DALI_CORE_TRAITS_H_
#define DALI_CORE_TRAITS_H_

#include <type_traits>
#include <array>
#include <vector>
#include "dali/core/host_dev.h"

namespace dali {

template <typename T>
struct is_vector : std::false_type {};

template <typename T, typename A>
struct is_vector<std::vector<T, A> > : std::true_type {};

template <typename T>
struct is_std_array : std::false_type {};

template <typename T, size_t A>
struct is_std_array<std::array<T, A> > : std::true_type {};

template <typename T>
using remove_const_t = std::remove_const_t<T>;

template <typename T>
using remove_cv_t = std::remove_cv_t<T>;

template <bool Value, typename Type = void>
using enable_if_t = std::enable_if_t<Value, Type>;


// check if the type has `resize` function callable with given arguments

template <typename T, typename... ResizeArgs,
          typename resize_result = decltype(std::declval<T&>().resize(ResizeArgs()...))>
inline std::true_type HasResize(T *, ResizeArgs... args);

inline std::false_type HasResize(...);

/// @brief Inerits `true_type`, if `T::resize` can be called with given arguments
template <typename T, typename... ResizeArgs>
struct has_resize : decltype(HasResize((T*)0, ResizeArgs()...)) {};  // NOLINT

/// @brief Inerits `true_type`, `if T::resize` can be called with an integer
template <typename T>
struct has_resize<T> : decltype(HasResize((T*)0, 1)) {};  // NOLINT

DALI_NO_EXEC_CHECK
template <typename T, typename Size>
DALI_HOST_DEV
inline std::enable_if_t<has_resize<T>::value> resize_if_possible(T &object, Size size) {
  object.resize(size);
}

template <typename T, typename Size>
DALI_HOST_DEV
inline std::enable_if_t<!has_resize<T>::value> resize_if_possible(T &object, Size size) {}

}  // namespace dali

#endif  // DALI_CORE_TRAITS_H_
