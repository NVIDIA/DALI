// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/mm/cuda_vm_resource.h"

namespace dali {
namespace mm {
namespace test {

class VMResourceTest : public ::testing::Test {
 public:
  void TestAlloc() {
    cuda_vm_resource res;
    res.block_size_ = 32<<20;  // fix the block size at 32 MiB for this test
    void *ptr = res.allocate(100<<20);
    void *ptr2 = res.allocate(100<<20);
    EXPECT_EQ(res.va_regions_.size(), 1u);
    EXPECT_EQ(res.va_regions_.front().mapped.find(false), 7);
    res.deallocate(ptr, 100<<20);
    EXPECT_EQ(res.va_regions_.front().available.find(true), 0);
    EXPECT_EQ(res.va_regions_.front().available.find(false), 3);
    void *ptr3 = res.allocate(100<<20);
    EXPECT_EQ(ptr, ptr3);
    cuda_vm_resource::mem_handle_t blocks[3];
    for (int i = 0; i < 3; i++)
      blocks[i] = res.va_regions_.front().mapping[i];
    res.deallocate(ptr3, 100<<20);
    // let's request more than was deallocated - it should go at the end of VA range
    void *ptr4 = res.allocate(150<<20);
    EXPECT_EQ(ptr4, static_cast<void*>(static_cast<char*>(ptr2) + (100<<20)));
    for (int i = 0; i < 3; i++) {
      // ptr4 should start 200 MiB from start, which is block 6
      // block 7 should be still unmapped, hence i+7.
      // Let's check that the 1st 3 blocks have been reused rather than newly allocated.
      EXPECT_EQ(res.va_regions_.back().mapping[i+7], blocks[i]);
    }
    res.deallocate(ptr2, 100<<20);
    res.deallocate(ptr4, 150<<20);
  }

  void TestRegionExtend() {

  }
};

TEST_F(VMResourceTest, BasicTest) {
  this->TestAlloc();
}

TEST_F(VMResourceTest, RegionExtend) {
  this->TestRegionExtend();
}

}  // namespace test
}  // namespace mm
}  // namespace dali


