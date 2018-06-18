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
