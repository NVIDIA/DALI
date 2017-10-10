#include <benchmark/benchmark.h>

#include "ndll/common.h"
#include "ndll/pipeline/data/allocator.h"
#include "ndll/pipeline/data/caching_device_allocator.h"
#include "ndll/pipeline/init.h"

int main(int argc, char **argv) {
  // Try out the caching allocator wrappers
  ndll::CachingDeviceAllocator *gpu_allocator =
    new ndll::CachingDeviceAllocator(new ndll::GPUAllocator);
  
  ndll::NDLLInit(new ndll::PinnedCPUAllocator, gpu_allocator);
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
