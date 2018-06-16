// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/pipeline/init.h"

#include "dali/pipeline/data/backend.h"

namespace dali {

void DALIInit(const OpSpec &cpu_allocator,
              const OpSpec &pinned_cpu_allocator,
              const OpSpec &gpu_allocator) {
  InitializeBackends(cpu_allocator, pinned_cpu_allocator, gpu_allocator);
}

void DALISetCPUAllocator(const OpSpec& allocator) {
  SetCPUAllocator(allocator);
}

void DALISetPinnedCPUAllocator(const OpSpec& allocator) {
  SetPinnedCPUAllocator(allocator);
}

void DALISetGPUAllocator(const OpSpec& allocator) {
  SetGPUAllocator(allocator);
}

}  // namespace dali
