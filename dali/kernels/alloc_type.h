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

#ifndef DALI_KERNELS_ALLOC_TYPE_H_
#define DALI_KERNELS_ALLOC_TYPE_H_

#include "dali/core/backend_tags.h"

namespace dali {
namespace kernels {

enum class AllocType : uint8_t {
  Host = 0,
  Pinned,
  GPU,
  Unified,
  Count,
};

template <AllocType alloc>
struct alloc_to_backend {
  using type = StorageCPU;
};

template <>
struct alloc_to_backend<AllocType::GPU> {
  using type = StorageGPU;
};
template <>
struct alloc_to_backend<AllocType::Unified> {
  using type = StorageUnified;
};

template <AllocType alloc>
using AllocBackend = typename alloc_to_backend<alloc>::type;

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_ALLOC_TYPE_H_
