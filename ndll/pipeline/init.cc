#include "ndll/pipeline/init.h"

#include "ndll/pipeline/data/backend.h"

namespace ndll {

void NDLLInit(CPUAllocator *cpu_allocator, GPUAllocator *gpu_allocator) {
  InitializeBackends(cpu_allocator, gpu_allocator);
}

} // namespace ndll
