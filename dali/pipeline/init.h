// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef DALI_PIPELINE_INIT_H_
#define DALI_PIPELINE_INIT_H_

#include "dali/pipeline/data/allocator.h"

namespace dali {

/**
 * @brief Initializes the pipeline. Sets global cpu&gpu allocators for all
 * pipeline objects. Must be called prior to constructing pipeline objects.
 * This must be called only once within a process.
 */
void DALIInit(const OpSpec &cpu_allocator,
              const OpSpec &pinned_cpu_allocator,
              const OpSpec &gpu_allocator);

void DALISetCPUAllocator(const OpSpec& allocator);
void DALISetPinnedCPUAllocator(const OpSpec& allocator);
void DALISetGPUAllocator(const OpSpec& allocator);


}  // namespace dali

#endif  // DALI_PIPELINE_INIT_H_
