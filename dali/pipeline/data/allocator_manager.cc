// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/pipeline/data/allocator_manager.h"

#include <map>
#include <memory>
#include <mutex>

#include "dali/pipeline/data/global_workspace.h"

namespace dali {

unique_ptr<CPUAllocator> AllocatorManager::cpu_allocator_(nullptr);
unique_ptr<CPUAllocator> AllocatorManager::pinned_cpu_allocator_(nullptr);
unique_ptr<GPUAllocator> AllocatorManager::gpu_allocator_(nullptr);
std::mutex AllocatorManager::mutex_;

}  // namespace dali
