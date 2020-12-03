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

#ifndef DALI_CORE_MM_POOL_RESOURCE_H_
#define DALI_CORE_MM_POOL_RESOURCE_H_

#include <mutex>
#include "dali/core/mm/memory_resource.h"
#include "dali/core/mm/detail/free_list.h"
#include "dali/core/small_vector.h"

namespace dali {
namespace mm {

template <typename FreeList, typename LockType>
class stream_aware_pool : public stream_memory_resource {
  using stream_event = std::pair<EventLeaseF, cudaStream_t>
};

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_POOL_RESOURCE_H_
