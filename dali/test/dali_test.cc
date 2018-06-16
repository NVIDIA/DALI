// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include <gtest/gtest.h>

#include "dali/pipeline/data/allocator.h"
#include "dali/pipeline/init.h"
#include "dali/pipeline/operators/op_spec.h"

int main(int argc, char **argv) {
  dali::DALIInit(dali::OpSpec("CPUAllocator"),
      dali::OpSpec("PinnedCPUAllocator"),
      dali::OpSpec("GPUAllocator"));
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
