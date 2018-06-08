// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include <gtest/gtest.h>

#include "ndll/pipeline/data/allocator.h"
#include "ndll/pipeline/init.h"
#include "ndll/pipeline/operators/op_spec.h"

int main(int argc, char **argv) {
  ndll::NDLLInit(ndll::OpSpec("CPUAllocator"),
      ndll::OpSpec("PinnedCPUAllocator"),
      ndll::OpSpec("GPUAllocator"));
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
