// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/benchmark/dali_bench.h"

#include <benchmark/benchmark.h>

#include "dali/common.h"
#include "dali/pipeline/data/allocator.h"
#include "dali/pipeline/init.h"
#include "dali/pipeline/operators/op_spec.h"

int main(int argc, char **argv) {
  dali::DALIInit(dali::OpSpec("CPUAllocator"),
      dali::OpSpec("PinnedCPUAllocator"),
      dali::OpSpec("GPUAllocator"));
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
