// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_DATA_ALLOCATOR_MANAGER_H_
#define DALI_PIPELINE_DATA_ALLOCATOR_MANAGER_H_

#include <memory>
#include <mutex>

#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/data/allocator.h"

namespace dali {

class AllocatorManager {
 public:
  AllocatorManager() {}
  ~AllocatorManager() = default;

  CPUAllocator& GetCPUAllocator() const {
    DALI_ENFORCE(cpu_allocator_ != nullptr,
        "DALI CPU allocator not set. Did you forget to call DALIInit?");
    return *cpu_allocator_.get();
  }

  CPUAllocator& GetPinnedCPUAllocator() const {
    DALI_ENFORCE(pinned_cpu_allocator_ != nullptr,
        "DALI Pinned CPU allocator not set. Did you forget to call DALIInit?");
    return *pinned_cpu_allocator_.get();
  }

  GPUAllocator& GetGPUAllocator() const {
    DALI_ENFORCE(gpu_allocator_ != nullptr,
        "DALI GPU allocator not set. Did you forget to call DALIInit?");
    return *gpu_allocator_.get();
  }

  void SetCPUAllocator(const OpSpec& allocator) {
    cpu_allocator_ = CPUAllocatorRegistry::Registry()
      .Create(allocator.name(), allocator);
  }

  void SetPinnedCPUAllocator(const OpSpec& allocator) {
    pinned_cpu_allocator_ = CPUAllocatorRegistry::Registry()
      .Create(allocator.name(), allocator);
  }

  void SetGPUAllocator(const OpSpec& allocator) {
    gpu_allocator_ = GPUAllocatorRegistry::Registry()
      .Create(allocator.name(), allocator);
  }

 private:
  shared_ptr<CPUAllocator> cpu_allocator_;
  shared_ptr<CPUAllocator> pinned_cpu_allocator_;
  shared_ptr<GPUAllocator> gpu_allocator_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_ALLOCATOR_MANAGER_H_
