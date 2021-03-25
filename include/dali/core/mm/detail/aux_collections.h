// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_CORE_MM_DETAIL_AUX_COLLECTIONS_H_
#define DALI_CORE_MM_DETAIL_AUX_COLLECTIONS_H_

#include <map>
#include <set>
#include <functional>
#include <utility>
#include "dali/core/mm/detail/aux_alloc.h"

namespace dali {
namespace mm {
namespace detail {

template <typename Key,
          typename Value,
          bool thread_local_pool = false,
          typename Cmp = std::less<Key>>
using pooled_map = std::map<
    Key, Value, Cmp, object_pool_allocator<std::pair<const Key, Value>, thread_local_pool>>;

template <typename T, bool thread_local_pool = false, typename Cmp = std::less<T>>
using pooled_set = std::set<
    T, Cmp, object_pool_allocator<T, thread_local_pool>>;

}  // namespace detail
}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_DETAIL_AUX_COLLECTIONS_H_
