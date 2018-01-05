// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/data/allocator.h"

namespace ndll {

// Define the CPU & GPU allocator registries
NDLL_DEFINE_OPTYPE_REGISTRY(GPUAllocator, GPUAllocator);
NDLL_DEFINE_OPTYPE_REGISTRY(CPUAllocator, CPUAllocator);

// Register GPU, CPU, and PinnedCPU allocators
NDLL_REGISTER_GPU_ALLOCATOR(GPUAllocator, GPUAllocator);
NDLL_REGISTER_CPU_ALLOCATOR(CPUAllocator, CPUAllocator);
NDLL_REGISTER_CPU_ALLOCATOR(PinnedCPUAllocator, PinnedCPUAllocator);

}  // namespace ndll
