// Copyright (c) 2018-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_BACKEND_TAGS_H_
#define DALI_CORE_BACKEND_TAGS_H_

#include <type_traits>
#include <cuda/memory_resource>

namespace dali {

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

template<typename ComputeBackend>
struct compute_to_storage {
  using type = StorageCPU;
};

template<>
struct compute_to_storage<ComputeGPU> {
  using type = StorageGPU;
};

template<class ComputeBackend>
using compute_to_storage_t = typename compute_to_storage<ComputeBackend>::type;

template <typename MemoryKind>
struct kind2storage;

template <>
struct kind2storage<cuda::memory_kind::host> {
  using type = StorageCPU;
};

template <>
struct kind2storage<cuda::memory_kind::pinned> {
  using type = StorageCPU;
};

template <>
struct kind2storage<cuda::memory_kind::device> {
  using type = StorageGPU;
};

template <>
struct kind2storage<cuda::memory_kind::managed> {
  using type = StorageUnified;
};

template <typename MemoryKind>
using kind2storage_t = typename kind2storage<MemoryKind>::type;

}  // namespace dali

#endif  // DALI_CORE_BACKEND_TAGS_H_
