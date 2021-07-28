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
#include "dali/core/mm/malloc_resource.h"
#include "dali/core/mm/detail/align.h"

namespace dali {
namespace mm {
namespace test {

template <typename Resource>
struct MMBasicResourceTest : public ::testing::Test {
  void AllocateTest() {
    Resource res;
    void *p1 = res.allocate(100);
    EXPECT_NE(p1, nullptr);
    void *p2 = res.allocate(200);
    EXPECT_NE(p2, nullptr);
    void *p3 = res.allocate(200000);
    EXPECT_NE(p3, nullptr);

    EXPECT_NE(p1, p2);
    EXPECT_NE(p1, p3);
    EXPECT_NE(p2, p3);
    res.deallocate(p2, 200);
    res.deallocate(p1, 100);
    res.deallocate(p3, 200000);
  }

  void AlignmentTest() {
    Resource res;
    void *p1 = res.allocate(1<<20, 256);
    EXPECT_NE(p1, nullptr);
    EXPECT_TRUE(detail::is_aligned(p1, 256));
    void *p2 = res.allocate(13, 32);
    EXPECT_NE(p2, nullptr);
    EXPECT_TRUE(detail::is_aligned(p2, 32));
    res.deallocate(p1, 1<<20, 256);
    res.deallocate(p2, 13, 32);
  }

  void OOMTest() {
    Resource res;
    EXPECT_THROW((res.allocate(static_cast<size_t>(-1L) / 4)), std::bad_alloc);
    // TODO(michalz): Remove when error handling is fixed in RMM
    (void)cudaGetLastError();
  }
};

using BasicResourceTypes = ::testing::Types<
  malloc_memory_resource,
  cuda_malloc_memory_resource,
  pinned_malloc_memory_resource>;
TYPED_TEST_SUITE(MMBasicResourceTest, BasicResourceTypes);


TYPED_TEST(MMBasicResourceTest, Allocate) {
  this->AllocateTest();
}

TYPED_TEST(MMBasicResourceTest, Alignment) {
  this->AlignmentTest();
}

TYPED_TEST(MMBasicResourceTest, OOM) {
  this->OOMTest();
}

}  // namespace test
}  // namespace mm
}  // namespace dali
