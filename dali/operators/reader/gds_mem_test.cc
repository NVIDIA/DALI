// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include "dali/operators/reader/gds_mem.h"

namespace dali {
namespace gds {

TEST(GDSMem, AllocatorMultiDevice) {
  int ndev;
  CUDA_CALL(cudaGetDeviceCount(&ndev));
  if (ndev < 2) {
    GTEST_SKIP() << "This test requires more than one CUDA capable device to run.";
    return;
  }
  ASSERT_NE(&GDSAllocator::instance(0), &GDSAllocator::instance(1));
}

TEST(GDSMem, Allocator) {
  mm::uptr<uint8_t> unq = gds::gds_alloc_unique(1024);
  ASSERT_NE(unq, nullptr);
  CUDA_CALL(cudaMemset(unq.get(), 0, 1024));
  unq.reset();
}

}  // namespace gds
}  // namespace dali
