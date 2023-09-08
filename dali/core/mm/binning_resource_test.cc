// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <map>
#include <memory>
#include "dali/core/mm/binning_resource.h"
#include "dali/core/span.h"

namespace dali {
namespace mm {
namespace test {

struct DummyResource : public memory_resource<mm::memory_kind::host> {
  void *do_allocate(size_t bytes, size_t alignment) {
    (void)alignment;
    allocs_by_size_[bytes]++;
    return &allocs_by_size_[bytes];
  }

  void do_deallocate(void *ptr, size_t bytes, size_t alignment) {
    ASSERT_EQ(ptr, &allocs_by_size_[bytes]);
    deallocs_by_size_[bytes]++;
  }

  std::map<size_t, int> allocs_by_size_;
  std::map<size_t, int> deallocs_by_size_;
};

TEST(BinningResourceTest, FindBin) {
  std::vector<std::shared_ptr<DummyResource>> resources;
  for (int i = 0; i < 4; i++)
    resources.push_back(std::make_shared<DummyResource>());

  size_t bin_split_points[] =  { 32, 1024, 1<<20 };
  binning_resource binning(
    make_cspan(bin_split_points),
    make_cspan(resources));

  void *p0 = binning.allocate(1);
  binning.deallocate(p0, 1);
  void *p1 = binning.allocate(32);
  binning.deallocate(p1, 32);
  void *p2 = binning.allocate(256);
  binning.deallocate(p2, 256);
  void *p3 = binning.allocate(1024);
  binning.deallocate(p3, 1024);
  void *p4 = binning.allocate(65536);
  binning.deallocate(p4, 65536);
  void *p5 = binning.allocate(1<<20);
  binning.deallocate(p5, 1<<20);
  void *p6 = binning.allocate(10<<20);
  binning.deallocate(p6, 10<<20);
  EXPECT_EQ(resources[0]->allocs_by_size_[1], 1);
  EXPECT_EQ(resources[0]->deallocs_by_size_[1], 1);
  EXPECT_EQ(resources[0]->allocs_by_size_[32], 1);
  EXPECT_EQ(resources[0]->deallocs_by_size_[32], 1);
  EXPECT_EQ(resources[1]->allocs_by_size_[256], 1);
  EXPECT_EQ(resources[1]->deallocs_by_size_[256], 1);
  EXPECT_EQ(resources[1]->allocs_by_size_[1024], 1);
  EXPECT_EQ(resources[1]->deallocs_by_size_[1024], 1);
  EXPECT_EQ(resources[2]->allocs_by_size_[65536], 1);
  EXPECT_EQ(resources[2]->deallocs_by_size_[65536], 1);
  EXPECT_EQ(resources[2]->allocs_by_size_[1<<20], 1);
  EXPECT_EQ(resources[2]->deallocs_by_size_[1<<20], 1);
  EXPECT_EQ(resources[3]->allocs_by_size_[10<<20], 1);
  EXPECT_EQ(resources[3]->deallocs_by_size_[10<<20], 1);
}

}  // namespace test
}  // namespace mm
}  // namespace dali

