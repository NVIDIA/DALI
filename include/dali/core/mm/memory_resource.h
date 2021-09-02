// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_MM_MEMORY_RESOURCE_H_
#define DALI_CORE_MM_MEMORY_RESOURCE_H_

#include <cuda_runtime.h>
#include <cstddef>

#include <cuda/memory_resource>

namespace dali {

/**
 * @brief Memory Manager
 *
 * This namespace contains classes and function for memory managment.
 * DALI memory manager follows the interface of C++17 polymorphic memory resource for
 * ordinary host allocators and extends them by the CUDA stream support for
 * stream-aware allocators.
 *
 * Some of the memory resources are composable, accepting an upstream memory resource.
 * Such composite resources can be used to quickly build an application-specific memory resource,
 * tailored to specific needs.
 */
namespace mm {

namespace memory_kind = cuda::memory_kind;

using cuda::any_context;
using cuda::memory_resource;
using cuda::resource_view;
using cuda::stream_ordered_resource_view;

using host_memory_resource = memory_resource<memory_kind::host>;
using pinned_memory_resource = memory_resource<memory_kind::pinned>;
using cuda::stream_view;

template <typename Kind>
using async_memory_resource = cuda::stream_ordered_memory_resource<Kind>;

using device_async_resource = async_memory_resource<memory_kind::device>;
using pinned_async_resource = async_memory_resource<memory_kind::pinned>;
using managed_async_resource = async_memory_resource<memory_kind::managed>;

struct stream_context {
  stream_view stream;
};

namespace detail {

template <typename Kind>
constexpr bool is_host_accessible = cuda::kind_has_property<Kind, cuda::memory_access::host>::value;

}  // namespace detail

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_MEMORY_RESOURCE_H_
