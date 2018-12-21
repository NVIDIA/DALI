// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include <vector>
#include <array>
#include <cassert>
#include "dali/kernels/kernel_req.h"

namespace dali {
namespace kernels {

static_assert(align_up(0, 2) == 0, "0 aligned up to 2 is 0");
static_assert(align_up(1, 2) == 2, "1 aligned up to 2 is 2");
static_assert(align_up(2, 2) == 2, "2 aligned up to 2 is 2");
static_assert(align_up(0, 8) == 0, "0 aligned up to 8 is 0");
static_assert(align_up(1, 8) == 8, "1 aligned up to 8 is 8");
static_assert(align_up(2, 8) == 8, "2 aligned up to 8 is 8");
static_assert(align_up(3, 8) == 8, "3 aligned up to 8 is 8");
static_assert(align_up(4, 8) == 8, "4 aligned up to 8 is 8");
static_assert(align_up(5, 8) == 8, "5 aligned up to 8 is 8");
static_assert(align_up(6, 8) == 8, "6 aligned up to 8 is 8");
static_assert(align_up(7, 8) == 8, "7 aligned up to 8 is 8");
static_assert(align_up(8, 8) == 8, "8 aligned up to 8 is 8");
static_assert(align_up(9, 8) == 16, "9 aligned up to 8 is 16");

template <typename T>
void test_add(ScratchpadEstimator &E, AllocType type, size_t count, size_t align = alignof(T)) {
  size_t prev = E.sizes[static_cast<int>(type)];
  EXPECT_EQ(align&(align-1), 0) << "Alignment must be a power of 2";
  size_t base = align_up(prev, align);
  E.add<T>(type, count, align);
  EXPECT_EQ(E.sizes[static_cast<int>(type)], base + count*sizeof(T));
}

TEST(Scratch, Estimator) {
  ScratchpadEstimator E;
  test_add<float>(E, AllocType::Host, 9);
  test_add<char>(E, AllocType::Host, 1);
  test_add<char>(E, AllocType::Host, 1);
  test_add<double>(E, AllocType::Host, 2);

  test_add<char>(E, AllocType::GPU, 1);
  test_add<float>(E, AllocType::GPU, 9);
  test_add<char>(E, AllocType::GPU, 1);
  test_add<double>(E, AllocType::GPU, 2);
  EXPECT_EQ(E.sizes[static_cast<int>(AllocType::Host)], 56);
  EXPECT_EQ(E.sizes[static_cast<int>(AllocType::GPU)], 64);
}

}  // namespace kernels
}  // namespace dali
