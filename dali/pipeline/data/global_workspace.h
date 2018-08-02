// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef DALI_PIPELINE_DATA_GLOBAL_WORKSPACE_H_
#define DALI_PIPELINE_DATA_GLOBAL_WORKSPACE_H_

#include <map>
#include <memory>
#include <set>
#include <vector>

#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/data/allocator_manager.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/buffer_manager.h"

namespace dali {

class GlobalWorkspace {
 public:
  static GlobalWorkspace *Get();
  static GlobalWorkspace *Get(int device);

  ~GlobalWorkspace() {
    return;
  }

  // allocator stuff
  void SetAllocators(const OpSpec &cpu_allocator,
                     const OpSpec &pinned_cpu_allocator,
                     const OpSpec &gpu_allocator) {
    AllocatorManager::SetCPUAllocator(cpu_allocator);
    AllocatorManager::SetPinnedCPUAllocator(pinned_cpu_allocator);
    AllocatorManager::SetGPUAllocator(gpu_allocator);
  }

  // set the indidivual allocators
  void SetCPUAllocator(const OpSpec& allocator) {
    AllocatorManager::SetCPUAllocator(allocator);
  }
  void SetPinnedCPUAllocator(const OpSpec& allocator) {
    AllocatorManager::SetPinnedCPUAllocator(allocator);
  }
  void SetGPUAllocator(const OpSpec& allocator) {
    AllocatorManager::SetGPUAllocator(allocator);
  }

  // get the individual allocators
  CPUAllocator& GetCPUAllocator() {
    return AllocatorManager::GetCPUAllocator();
  }
  CPUAllocator& GetPinnedCPUAllocator() {
    return AllocatorManager::GetPinnedCPUAllocator();
  }
  GPUAllocator& GetGPUAllocator() {
    return AllocatorManager::GetGPUAllocator();
  }

  // Actual allocations
  void *AllocateHost(const size_t bytes, const bool pinned) {
    void *ptr = nullptr;
    if (!pinned) {
      GetCPUAllocator().New(&ptr, bytes);
    } else {
      GetPinnedCPUAllocator().New(&ptr, bytes);
    }
    return ptr;
  }

  void FreeHost(void *ptr, const size_t bytes, const bool pinned) {
    if (!pinned) {
      GetCPUAllocator().Delete(ptr, bytes);
    } else {
      GetPinnedCPUAllocator().Delete(ptr, bytes);
    }
  }

  void *AllocateGPU(const size_t bytes, const bool pinned = false);
  void FreeGPU(void *ptr, const size_t bytes, const bool pinned = false);

  template <typename Backend>
  unique_ptr<Buffer<Backend>> AcquireBuffer(size_t size, bool pinned);

  template <typename Backend>
  void ReleaseBuffer(unique_ptr<Buffer<Backend>> *buffer, bool pinned);

 private:
  explicit GlobalWorkspace(int device);

  int device_;

  // Buffer manager
  unique_ptr<BufferManagerBase> buffer_manager_;
};

template <>
inline unique_ptr<Buffer<CPUBackend>> GlobalWorkspace::AcquireBuffer(size_t size, bool pinned) {
  return buffer_manager_->AcquireBuffer(size, pinned);
}

template <>
inline unique_ptr<Buffer<GPUBackend>> GlobalWorkspace::AcquireBuffer(size_t size, bool pinned) {
  return buffer_manager_->AcquireBuffer(size);
}

template <>
inline void GlobalWorkspace::ReleaseBuffer(unique_ptr<Buffer<CPUBackend>> *buffer, bool pinned) {
  buffer_manager_->ReleaseBuffer(buffer, pinned);
}

template <>
inline void GlobalWorkspace::ReleaseBuffer(unique_ptr<Buffer<GPUBackend>> *buffer, bool pinned) {
  buffer_manager_->ReleaseBuffer(buffer);
}

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_GLOBAL_WORKSPACE_H_
