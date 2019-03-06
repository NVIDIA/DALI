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

#ifndef DALI_KERNELS_TYPE_TAG_H_
#define DALI_KERNELS_TYPE_TAG_H_

#include <type_traits>

namespace dali {

template <typename T,
  bool integral = std::is_integral<T>::value,
  bool fp = std::is_floating_point<T>::value>
struct TypeTagBase;

template <typename T>
struct TypeTagBase<T, true, false>
 : std::integral_constant<int, sizeof(T) | (std::is_unsigned<T>::value << 8) > {};

template <typename T>
struct TypeTagBase<T, false, true>
 : std::integral_constant<int, sizeof(T) | (1<<10) > {};

template <typename T>
struct TypeTag : TypeTagBase<T> {};

}  // namespace dali

#endif  // DALI_KERNELS_TYPE_TAG_H_
