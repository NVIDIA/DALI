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

#include "dali/pipeline/data/backend.h"

#include <atomic>
#include <mutex>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/pipeline/data/allocator.h"
#include "dali/pipeline/operator/op_spec.h"

namespace dali {

class AllocatorManager {
 public:
  AllocatorManager(): cpu_allocator_{nullptr}, pinned_cpu_allocator_{nullptr},
                      gpu_allocators_(nullptr) {
  }

  ~AllocatorManager() {
    CPUAllocator* null_cpu_allocator = nullptr;
    GPUAllocator* null_gpu_allocator = nullptr;
    std::vector<std::atomic<GPUAllocator*>>* null_alloc_vect = nullptr;
    delete std::atomic_exchange(&cpu_allocator_, null_cpu_allocator);
    delete std::atomic_exchange(&pinned_cpu_allocator_, null_cpu_allocator);
    auto gpu_alloc_vect = std::atomic_exchange(&gpu_allocators_, null_alloc_vect);
    if (gpu_alloc_vect) {
      for (auto& gpu_alloc : *gpu_alloc_vect) {
        delete std::atomic_exchange(&gpu_alloc, null_gpu_allocator);
      }
      delete gpu_alloc_vect;
    }
  }

  void SetAllocators(const OpSpec &cpu_allocator,
                     const OpSpec &pinned_cpu_allocator,
                     const OpSpec &gpu_allocator) {
    DALI_ENFORCE(cpu_allocator_ == nullptr, "DALI CPU allocator already set");
    DALI_ENFORCE(pinned_cpu_allocator_ == nullptr, "DALI Pinned CPU allocator already set");
    DALI_ENFORCE(!gpu_opspec_, "DALI GPU allocator already set");
    SetCPUAllocator(cpu_allocator);
    SetPinnedCPUAllocator(pinned_cpu_allocator);
    gpu_opspec_.reset(new OpSpec(gpu_allocator));
    // Don't set GPU allocator, it will be done in the first use
  }

  CPUAllocator& GetCPUAllocator() {
    DALI_ENFORCE(cpu_allocator_ != nullptr,
        "DALI CPU allocator not set. Did you forget to call DALIInit?");
    return *cpu_allocator_;
  }

  CPUAllocator& GetPinnedCPUAllocator() {
    DALI_ENFORCE(cpu_allocator_ != nullptr,
        "DALI Pinned CPU allocator not set. Did you forget to call DALIInit?");
    return *pinned_cpu_allocator_;
  }

  GPUAllocator& GetGPUAllocator() {
    int dev;
    CUDA_CALL(cudaGetDevice(&dev));
    auto gpu_alloc_vect = GetGPUAllocatorsVect();
    auto alloc = (*gpu_alloc_vect)[dev].load();
    if (alloc) {
      return *alloc;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    alloc = (*gpu_alloc_vect)[dev].load();
    if (!alloc) {
      SetGPUAllocator(*gpu_opspec_);
    }
    return *(*gpu_alloc_vect)[dev].load();
  }

  void SetCPUAllocator(const OpSpec& allocator) {
    auto alloc = CPUAllocatorRegistry::Registry().Create(allocator.name(), allocator).release();
    delete std::atomic_exchange(&cpu_allocator_, alloc);
  }

  void SetPinnedCPUAllocator(const OpSpec& allocator) {
    auto alloc = CPUAllocatorRegistry::Registry().Create(allocator.name(), allocator).release();
    delete std::atomic_exchange(&pinned_cpu_allocator_, alloc);
  }

  void SetGPUAllocator(const OpSpec& allocator) {
    auto alloc = GPUAllocatorRegistry::Registry().Create(allocator.name(), allocator);
    SetGPUAllocator(std::move(alloc));
  }

  void SetGPUAllocator(std::unique_ptr<GPUAllocator> allocator) {
    int dev;
    CUDA_CALL(cudaGetDevice(&dev));
    delete std::atomic_exchange(&((*GetGPUAllocatorsVect())[dev]), allocator.release());
  }

 private:
  std::vector<std::atomic<GPUAllocator*>> *GetGPUAllocatorsVect() {
    auto gpu_alloc_vect = gpu_allocators_.load();
    if (!gpu_alloc_vect) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!gpu_allocators_.load()) {
        int dev_cout = 0;
        CUDA_CALL(cudaGetDeviceCount(&dev_cout));
        auto new_vect = new std::vector<std::atomic<GPUAllocator*>>(dev_cout);
        delete std::atomic_exchange(&gpu_allocators_, new_vect);
      }
      gpu_alloc_vect = gpu_allocators_.load();
    }
    return gpu_alloc_vect;
  }
  std::atomic<CPUAllocator*> cpu_allocator_;
  std::atomic<CPUAllocator*> pinned_cpu_allocator_;
  std::atomic<std::vector<std::atomic<GPUAllocator*>>*> gpu_allocators_;
  unique_ptr<const OpSpec> gpu_opspec_;

  std::mutex mutex_;
};

static AllocatorManager allocator_mgr;

// Sets the allocator ptrs for all backends
void InitializeBackends(const OpSpec &cpu_allocator,
    const OpSpec &pinned_cpu_allocator,
    const OpSpec &gpu_allocator) {
  allocator_mgr.SetAllocators(cpu_allocator, pinned_cpu_allocator, gpu_allocator);
}

void SetCPUAllocator(const OpSpec& allocator) {
  allocator_mgr.SetCPUAllocator(allocator);
}

void SetPinnedCPUAllocator(const OpSpec& allocator) {
  allocator_mgr.SetPinnedCPUAllocator(allocator);
}

void SetGPUAllocator(const OpSpec& allocator) {
  allocator_mgr.SetGPUAllocator(allocator);
}

void SetGPUAllocator(std::unique_ptr<GPUAllocator> allocator) {
  allocator_mgr.SetGPUAllocator(std::move(allocator));
}

GPUAllocator& GetGPUAllocator() {
  return allocator_mgr.GetGPUAllocator();
}

void* GPUBackend::New(size_t bytes, bool) {
  void *ptr = nullptr;
  allocator_mgr.GetGPUAllocator().New(&ptr, bytes);
  return ptr;
}

void GPUBackend::Delete(void *ptr, size_t bytes, bool) {
  allocator_mgr.GetGPUAllocator().Delete(ptr, bytes);
}

void* CPUBackend::New(size_t bytes, bool pinned) {
  void *ptr = nullptr;
  if (!pinned) {
    allocator_mgr.GetCPUAllocator().New(&ptr, bytes);
  } else {
    allocator_mgr.GetPinnedCPUAllocator().New(&ptr, bytes);
  }
  return ptr;
}

void CPUBackend::Delete(void *ptr, size_t bytes, bool pinned) {
  if (!pinned) {
    allocator_mgr.GetCPUAllocator().Delete(ptr, bytes);
  } else {
    allocator_mgr.GetPinnedCPUAllocator().Delete(ptr, bytes);
  }
}

}  // namespace dali
