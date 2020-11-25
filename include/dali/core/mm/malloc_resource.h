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

#ifndef DALI_CORE_MM_MALLOC_RESOURCE_H_
#define DALI_CORE_MM_MALLOC_RESOURCE_H_

#include <stdlib.h>
#include "dali/core/mm/memory_resource.h"

namespace dali {
namespace mm {

/**
 * @brief A memory resource that manages host memory with std::aligned_alloc and std::free
 */
class malloc_memory_resource : public memory_resource {
  void *do_allocate(size_t bytes, size_t alignment) override {
    return aligned_alloc(alignment, bytes + sizeof(int));
  }

  void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
    return free(ptr);
  }

  bool do_is_equal(const memory_resource &other) const noexcept override {
    return dynamic_cast<const malloc_memory_resource*>(&other) != nullptr;
  }

 public:
  static malloc_memory_resource &instance() {
    static malloc_memory_resource inst;
    return inst;
  }
};

}  // namespace mm
}  // namespace dali


#endif  // DALI_CORE_MM_MALLOC_RESOURCE_H_
