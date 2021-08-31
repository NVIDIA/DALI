// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_MM_MEMORY_KIND_H_
#define DALI_CORE_MM_MEMORY_KIND_H_

#include <type_traits>
#include "dali/core/mm/memory_resource.h"
#include "dali/core/backend_tags.h"

namespace dali {
namespace mm {

enum memory_kind_id {
  host = 0,
  pinned,
  device,
  managed,
  count,
};

template <typename Kind>
struct kind2id;

template <typename Kind>
constexpr auto kind2id_v = kind2id<Kind>::value;

template <memory_kind_id id>
struct id2kind_helper;

template <memory_kind_id id>
using id2kind = typename id2kind_helper<id>::type;

#define DALI_MAP_MEM_KIND(Lind) \
template <> \
struct kind2id<memory_kind::Kind>
: std::integral_constant<memory_kind_id, memory_kind_id::Kind> {}; \
template <> \
struct kind2id<memory_kind_id::Kind> { \
  using type = memory_kind::Kind; \
}

DALI_MAP_MEM_KIND(host)
DALI_MAP_MEM_KIND(pinned)
DALI_MAP_MEM_KIND(device)
DALI_MAP_MEM_KIND(managed)

template <typename Kind>
struct kind2backend_helper {

};

template <typename Kind>
struct kind2backend_helper;

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_MEMORY_H_
