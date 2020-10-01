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

#ifndef DALI_KERNELS_COMMON_TYPE_ERASURE_H_
#define DALI_KERNELS_COMMON_TYPE_ERASURE_H_

#include <cuda_runtime.h>
#include <cstdint>

namespace dali {
namespace kernels {

template <int size>
struct type_of_size_helper;

template <>
struct type_of_size_helper<1> : same_as<int8_t> {};

template <>
struct type_of_size_helper<2> : same_as<int16_t> {};

template <>
struct type_of_size_helper<4> : same_as<int32_t> {};

template <>
struct type_of_size_helper<8> : same_as<int64_t> {};

template <>
struct type_of_size_helper<16> : same_as<int4> {};

template <int size>
using type_of_size = typename type_of_size_helper<size>::type;

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_TYPE_ERASURE_H_
