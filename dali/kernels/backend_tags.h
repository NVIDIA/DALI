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

#ifndef DALI_KERNELS_BACKEND_TAGS_H_
#define DALI_KERNELS_BACKEND_TAGS_H_

#include <type_traits>

namespace dali {
namespace kernels {

struct StorageGPU {};
struct StorageCPU {};
struct StorageUnified {};

struct ComputeGPU {};
struct ComputeCPU {};

template <typename Storage>
struct is_gpu_accessible : std::false_type {};

template <typename Storage>
struct is_cpu_accessible : std::false_type {};

template <>
struct is_gpu_accessible<StorageGPU> : std::true_type {};
template <>
struct is_gpu_accessible<StorageUnified> : std::true_type {};
template <>
struct is_cpu_accessible<StorageCPU> : std::true_type {};
template <>
struct is_cpu_accessible<StorageUnified> : std::true_type {};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_BACKEND_TAGS_H_
