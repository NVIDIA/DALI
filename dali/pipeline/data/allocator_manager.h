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
  static void SetAllocators(const OpSpec &cpu_allocator,
                            const OpSpec &pinned_cpu_allocator,
                            const OpSpec &gpu_allocator) {
    // Lock so we can give a good error if the user calls this from multiple threads.
    std::lock_guard<std::mutex> lock(mutex_);
    DALI_ENFORCE(cpu_allocator_ == nullptr, "DALI CPU allocator already set");
    DALI_ENFORCE(pinned_cpu_allocator_ == nullptr, "DALI Pinned CPU allocator already set");
    DALI_ENFORCE(gpu_allocator_ == nullptr, "DALI GPU allocator already set");
    cpu_allocator_ = CPUAllocatorRegistry::Registry()
      .Create(cpu_allocator.name(), cpu_allocator);
    pinned_cpu_allocator_ = CPUAllocatorRegistry::Registry()
      .Create(pinned_cpu_allocator.name(), pinned_cpu_allocator);
    gpu_allocator_ = GPUAllocatorRegistry::Registry()
      .Create(gpu_allocator.name(), gpu_allocator);
  }

  static CPUAllocator& GetCPUAllocator() {
    DALI_ENFORCE(cpu_allocator_ != nullptr,
        "DALI CPU allocator not set. Did you forget to call DALIInit?");
    return *cpu_allocator_.get();
  }

  static CPUAllocator& GetPinnedCPUAllocator() {
    DALI_ENFORCE(cpu_allocator_ != nullptr,
        "DALI Pinned CPU allocator not set. Did you forget to call DALIInit?");
    return *pinned_cpu_allocator_.get();
  }

  static GPUAllocator& GetGPUAllocator() {
    DALI_ENFORCE(gpu_allocator_ != nullptr,
        "DALI GPU allocator not set. Did you forget to call DALIInit?");
    return *gpu_allocator_.get();
  }

  static void SetCPUAllocator(const OpSpec& allocator) {
    std::lock_guard<std::mutex> lock(mutex_);
    cpu_allocator_ = CPUAllocatorRegistry::Registry()
      .Create(allocator.name(), allocator);
  }

  static void SetPinnedCPUAllocator(const OpSpec& allocator) {
    std::lock_guard<std::mutex> lock(mutex_);
    pinned_cpu_allocator_ = CPUAllocatorRegistry::Registry()
      .Create(allocator.name(), allocator);
  }

  static void SetGPUAllocator(const OpSpec& allocator) {
    std::lock_guard<std::mutex> lock(mutex_);
    gpu_allocator_ = GPUAllocatorRegistry::Registry()
      .Create(allocator.name(), allocator);
  }

 private:
  // AllocatorManager should be accessed through its static members
  AllocatorManager() {}

  static unique_ptr<CPUAllocator> cpu_allocator_;
  static unique_ptr<CPUAllocator> pinned_cpu_allocator_;
  static unique_ptr<GPUAllocator> gpu_allocator_;
  static std::mutex mutex_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_ALLOCATOR_MANAGER_H_
