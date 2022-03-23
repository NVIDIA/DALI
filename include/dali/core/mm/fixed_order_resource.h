// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_MM_FIXED_ORDER_RESOURCE_H_
#define DALI_CORE_MM_FIXED_ORDER_RESOURCE_H_

#include <dali/core/mm/memory_resource.h>
#include <dali/core/access_order.h>

namespace dali {
namespace mm {

/**
 * @brief A proxy resource that performs all allocations and deallocation in a fixed order
 *        (stream or host) and exposes a non-stream allocation API.
 *
 * NOTE: This class cannot be used by an arbitrary consumer of memory resources if the
 *       allocation order is not AccessOrder::host.
 *
 * NOTE: If this resource is used as an upstream for another resource, then the pool
 *       inherits the order of allocation and deallocation if:
 *       - the allocaton and deallocation order matches
 *       - it does not attempt to access the memory in order other than specified
 */
template <typename MemoryKind, typename Upstream = async_memory_resource<MemoryKind> >
class fixed_order_resource final : public memory_resource<MemoryKind> {
 public:
  fixed_order_resource() = default;

  fixed_order_resource(Upstream *upstream, AccessOrder order)
  : upstream_(upstream), alloc_order_(order), dealloc_order_(order) {}

  fixed_order_resource(Upstream *upstream, AccessOrder alloc_order, AccessOrder dealloc_order)
  : upstream_(upstream), alloc_order_(alloc_order), dealloc_order_(dealloc_order) {}

  AccessOrder alloc_order() const { return alloc_order_; }
  AccessOrder dealloc_order() const { return dealloc_order_; }

 private:
  void *do_allocate(size_t size, size_t alignment) final {
    if (alloc_order_.is_device())
      return upstream_->allocate_async(size, alignment, alloc_order_.stream());
    else
      return upstream_->allocate(size, alignment);
  }

  void do_deallocate(void *ptr, size_t size, size_t alignment) final {
    if (dealloc_order_.is_device())
      upstream_->deallocate_async(ptr, size, alignment, dealloc_order_.stream());
    else
      upstream_->deallocate(ptr, size, alignment);
  }

  Upstream *upstream_ = nullptr;
  AccessOrder alloc_order_, dealloc_order_;
};

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_FIXED_ORDER_RESOURCE_H_

