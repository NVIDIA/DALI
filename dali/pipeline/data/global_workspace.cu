// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/pipeline/data/global_workspace.h"

#include <memory>

#include "dali/pipeline/data/allocator.h"
#include "dali/pipeline/data/allocator_manager.h"

namespace dali {

void *GlobalWorkspace::AllocateGPU(const size_t bytes, const bool pinned) {
  void *ptr = nullptr;
  GetGPUAllocator().New(&ptr, bytes);

  return ptr;
}

void GlobalWorkspace::FreeGPU(void *ptr, const size_t bytes, const bool pinned) {
  GetGPUAllocator().Delete(ptr, bytes);
}

}  // namespace dali
