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

#include <memory>
#include <utility>
#include <vector>
#include "dali/operators/reader/gds_mem.h"
#include "dali/core/spinlock.h"
#include "dali/core/mm/pool_resource.h"
#include "dali/core/mm/malloc_resource.h"
#include "dali/core/mm/composite_resource.h"

namespace dali {

GDSAllocator::GDSAllocator() {
  // Currently, GPUDirect Storage can work only with memory allocated with cudaMalloc and
  // cuMemAlloc. Since DALI is transitioning to CUDA Virtual Memory Management for memory
  // allocation, we need a special allocator that's compatible with GDS.

  static auto upstream = std::make_shared<mm::cuda_malloc_memory_resource>();
  using resource_type = mm::pool_resource_base<
    mm::memory_kind::device, mm::coalescing_free_tree, spinlock>;
  auto rsrc = std::make_shared<resource_type>(upstream.get());
  rsrc_ = make_shared_composite_resource(std::move(rsrc), upstream);
}

GDSAllocator &GDSAllocator::instance(int device) {
  static int ndevs = []() {
    int devs = 0;
    CUDA_CALL(cudaGetDeviceCount(&devs));
    return devs;
  }();
  static vector<GDSAllocator> instances(ndevs);
  if (device < 0)
    CUDA_CALL(cudaGetDevice(&device));
  assert(device >= 0 && device < ndevs);
  return instances[device];
}

}  // namespace dali
