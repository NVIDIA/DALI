// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/data/backend.h"

#include <mutex>
#include <memory>
#include <unordered_map>
#include <utility>

#include "ndll/common.h"
#include "ndll/pipeline/data/allocator.h"
#include "ndll/pipeline/operators/op_spec.h"

namespace ndll {

#define MAP_INSERT_OR_UPDATE(VALUE)                     \
  int dev;                                              \
  CUDA_CALL(cudaGetDevice(&dev));                       \
  std::cout << "[BACKEND] " << dev << "\n";             \
  auto it = gpu_allocators_.find(dev);                  \
  if (it != gpu_allocators_.end()) {                    \
    std::cout << "VALUE updated" << endl;               \
    it->second = VALUE;                                 \
  } else {                                              \
    gpu_allocators_.insert(std::make_pair(dev, VALUE)); \
  }



class AllocatorManager {
 public:
  static void SetAllocators(const OpSpec &cpu_allocator,
                            const OpSpec &pinned_cpu_allocator,
                            const OpSpec &gpu_allocator) {
    // Lock so we can give a good error if the user calls this from multiple threads.
    std::lock_guard<std::mutex> lock(mutex_);
    NDLL_ENFORCE(cpu_allocator_ == nullptr, "NDLL CPU allocator already set");
    NDLL_ENFORCE(pinned_cpu_allocator_ == nullptr, "NDLL Pinned CPU allocator already set");
    NDLL_ENFORCE(gpu_allocators_.size() == 0, "NDLL GPU allocator already set");
    cpu_allocator_ = CPUAllocatorRegistry::Registry()
      .Create(cpu_allocator.name(), cpu_allocator);
    pinned_cpu_allocator_ = CPUAllocatorRegistry::Registry()
      .Create(pinned_cpu_allocator.name(), pinned_cpu_allocator);
    gpu_opspec_.reset(new OpSpec(gpu_allocator));
    MAP_INSERT_OR_UPDATE(
      GPUAllocatorRegistry::Registry().Create(gpu_allocator.name(), gpu_allocator));
  }

  static CPUAllocator& GetCPUAllocator() {
    NDLL_ENFORCE(cpu_allocator_ != nullptr,
        "NDLL CPU allocator not set. Did you forget to call NDLLInit?");
    return *cpu_allocator_.get();
  }

  static CPUAllocator& GetPinnedCPUAllocator() {
    NDLL_ENFORCE(cpu_allocator_ != nullptr,
        "NDLL Pinned CPU allocator not set. Did you forget to call NDLLInit?");
    return *pinned_cpu_allocator_.get();
  }

  static GPUAllocator& GetGPUAllocator() {
    int dev;
    CUDA_CALL(cudaGetDevice(&dev));
    auto gpu_allocator = gpu_allocators_.find(dev);
    // Lazy allocation per device
    if (gpu_allocator == gpu_allocators_.end()) {
      MAP_INSERT_OR_UPDATE(
        GPUAllocatorRegistry::Registry().Create(gpu_opspec_->name(), *gpu_opspec_));
      gpu_allocator = gpu_allocators_.find(dev);
    }
    return *gpu_allocator->second.get();
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
    MAP_INSERT_OR_UPDATE(
      GPUAllocatorRegistry::Registry().Create(allocator.name(), allocator));
  }

  static void SetGPUAllocator(std::unique_ptr<GPUAllocator> allocator) {
    std::lock_guard<std::mutex> lock(mutex_);
    MAP_INSERT_OR_UPDATE(std::move(allocator));
  }

 private:
  // AllocatorManager should be accessed through its static members
  AllocatorManager() {}

  static unique_ptr<CPUAllocator> cpu_allocator_;
  static unique_ptr<CPUAllocator> pinned_cpu_allocator_;
  static std::unordered_map<int, unique_ptr<GPUAllocator>> gpu_allocators_;
  static std::unique_ptr<const OpSpec> gpu_opspec_;

  static std::mutex mutex_;
};

unique_ptr<CPUAllocator> AllocatorManager::cpu_allocator_(nullptr);
unique_ptr<CPUAllocator> AllocatorManager::pinned_cpu_allocator_(nullptr);
std::unordered_map<int, unique_ptr<GPUAllocator>> AllocatorManager::gpu_allocators_;
unique_ptr<const OpSpec> AllocatorManager::gpu_opspec_(nullptr);
std::mutex AllocatorManager::mutex_;

// Sets the allocator ptrs for all backends
void InitializeBackends(const OpSpec &cpu_allocator,
    const OpSpec &pinned_cpu_allocator,
    const OpSpec &gpu_allocator) {
  AllocatorManager::SetAllocators(cpu_allocator, pinned_cpu_allocator, gpu_allocator);
}

void SetCPUAllocator(const OpSpec& allocator) {
  AllocatorManager::SetCPUAllocator(allocator);
}

void SetPinnedCPUAllocator(const OpSpec& allocator) {
  AllocatorManager::SetPinnedCPUAllocator(allocator);
}

void SetGPUAllocator(const OpSpec& allocator) {
  AllocatorManager::SetGPUAllocator(allocator);
}

void SetGPUAllocator(std::unique_ptr<GPUAllocator> allocator) {
  AllocatorManager::SetGPUAllocator(std::move(allocator));
}

GPUAllocator& GetGPUAllocator() {
  return AllocatorManager::GetGPUAllocator();
}

void* GPUBackend::New(size_t bytes, bool) {
  void *ptr = nullptr;
  AllocatorManager::GetGPUAllocator().New(&ptr, bytes);
  return ptr;
}

void GPUBackend::Delete(void *ptr, size_t bytes, bool) {
  AllocatorManager::GetGPUAllocator().Delete(ptr, bytes);
}

void* CPUBackend::New(size_t bytes, bool pinned) {
  void *ptr = nullptr;
  if (!pinned) {
    AllocatorManager::GetCPUAllocator().New(&ptr, bytes);
  } else {
    AllocatorManager::GetPinnedCPUAllocator().New(&ptr, bytes);
  }
  return ptr;
}

void CPUBackend::Delete(void *ptr, size_t bytes, bool pinned) {
  if (!pinned) {
    AllocatorManager::GetCPUAllocator().Delete(ptr, bytes);
  } else {
    AllocatorManager::GetPinnedCPUAllocator().Delete(ptr, bytes);
  }
}

}  // namespace ndll
