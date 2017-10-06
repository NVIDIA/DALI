#ifndef NDLL_PIPELINE_INIT_H_
#define NDLL_PIPELINE_INIT_H_

#include "ndll/pipeline/data/allocator.h"

namespace ndll {

/**
 * @brief Initializes the pipeline. Sets global cpu&gpu allocators for all
 * pipeline objects. Must be called prior to constructing pipeline objects.
 * This must be called only once within a process.
 */
void NDLLInit(CPUAllocator *cpu_allocator, GPUAllocator *gpu_allocator);


} // namespace ndll

#endif // NDLL_PIPELINE_INIT_H_
