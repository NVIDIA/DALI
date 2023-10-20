// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_MM_POOL_RESOURCE_BASE_H_
#define DALI_CORE_MM_POOL_RESOURCE_BASE_H_

#include "dali/core/mm/memory_resource.h"

namespace dali {
namespace mm {

template <typename Kind>
class pool_resource_base {
 public:
  virtual void release_unused() {}

  virtual void *try_allocate_from_free(size_t bytes, size_t alignment) {
    return nullptr;
  }
};

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_POOL_RESOURCE_BASE_H_
