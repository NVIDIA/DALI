// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/benchmark/ndll_bench.h"

#include <benchmark/benchmark.h>

#include "ndll/common.h"
#include "ndll/pipeline/data/allocator.h"
#include "ndll/pipeline/init.h"
#include "ndll/pipeline/op_spec.h"

int main(int argc, char **argv) {
  ndll::NDLLInit(ndll::OpSpec("CPUAllocator"),
      ndll::OpSpec("PinnedCPUAllocator"),
      ndll::OpSpec("GPUAllocator"));
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
