#include <benchmark/benchmark.h>

#include "ndll/common.h"
#include "ndll/pipeline/data/allocator.h"
#include "ndll/pipeline/init.h"

int main(int argc, char **argv) {
  ndll::NDLLInit(new ndll::PinnedCPUAllocator, new ndll::GPUAllocator);
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
