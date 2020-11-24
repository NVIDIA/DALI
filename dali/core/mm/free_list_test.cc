// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include <random>
#include "dali/core/mm/detail/free_list.h"

namespace dali {
namespace mm {

TEST(UniformFreeList, PutGet) {
  uniform_free_list fl;
  char a[2];
  fl.put(a);
  EXPECT_EQ(fl.get(), static_cast<void*>(a));
  EXPECT_EQ(fl.get(), nullptr);
  fl.put(a);
  fl.put(a+1);
  EXPECT_EQ(fl.get(), static_cast<void*>(a+1));
  EXPECT_EQ(fl.get(), static_cast<void*>(a+0));
  EXPECT_EQ(fl.get(), nullptr);
}

TEST(BestFitFreeList, PutGet) {
  best_fit_free_list fl;
  char a alignas(16)[1000];
  fl.put(a, 10);
  fl.put(a+10, 100);
  fl.put(a+110, 15);
  fl.put(a+125, 35);
  EXPECT_EQ(fl.get(10, 16), a);
  fl.put(a, 10);
  EXPECT_EQ(fl.get(8, 16), a);
  EXPECT_EQ(fl.get(2, 1), a+8);
  fl.put(a, 8);
  EXPECT_EQ(fl.get(100, 16), nullptr);
  EXPECT_EQ(fl.get(100, 2), a+10);
  EXPECT_EQ(fl.get(8, 16), a);
  EXPECT_EQ(fl.get(9, 16), a+112);
}

TEST(CoalescingFreeList, PutGet) {
  coalescing_free_list fl;
  char a alignas(16)[1000];
  // put some pieces and let the list coalesce
  fl.put(a, 10);
  fl.put(a+10, 100);
  fl.put(a+110, 15);
  fl.put(a+125, 35);
  // check if we can get a contiguous block
  EXPECT_EQ(fl.get(160, 16), a);
  // put it back in pieces
  fl.put(a+125, 35);
  fl.put(a+10, 100);
  fl.put(a, 10);
  fl.put(a+110, 15);

  // now some random stuff
  EXPECT_EQ(fl.get(10, 16), a);
  fl.put(a, 10);
  EXPECT_EQ(fl.get(8, 16), a);
  EXPECT_EQ(fl.get(2, 1), a+8);
  fl.put(a, 8);
  EXPECT_EQ(fl.get(8, 16), a);
  EXPECT_EQ(fl.get(100, 8), a+16);
  fl.put(a, 8);
  fl.put(a+8, 2);
  EXPECT_EQ(fl.get(9, 16), a);

  // put everything back again
  fl.put(a+16, 100);
  fl.put(a, 9);
  // and check coalescing
  EXPECT_EQ(fl.get(160, 16), a);
}

}  // namespace mm
}  // namespace dali
