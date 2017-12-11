// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/init.h"

#include "ndll/pipeline/data/backend.h"

namespace ndll {

void NDLLInit(const OpSpec &cpu_allocator, const OpSpec &gpu_allocator) {
  InitializeBackends(cpu_allocator, gpu_allocator);
}

}  // namespace ndll
