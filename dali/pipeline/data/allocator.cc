// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/pipeline/data/allocator.h"

namespace dali {

// Define the CPU & GPU allocator registries
DALI_DEFINE_OPTYPE_REGISTRY(GPUAllocator, GPUAllocator);
DALI_DEFINE_OPTYPE_REGISTRY(CPUAllocator, CPUAllocator);

// Register GPU, CPU, and PinnedCPU allocators
DALI_REGISTER_GPU_ALLOCATOR(GPUAllocator, GPUAllocator);
DALI_REGISTER_CPU_ALLOCATOR(CPUAllocator, CPUAllocator);
DALI_REGISTER_CPU_ALLOCATOR(PinnedCPUAllocator, PinnedCPUAllocator);

}  // namespace dali
