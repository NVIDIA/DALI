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
#include <vector>
#include "dali/core/tensor_shape.h"
#include "dali/kernels/common/split_shape.h"

namespace dali {
namespace kernels {

TEST(split_shape, basic_test) {
  TensorShape<> sh(10, 10, 10);
  std::vector<int> split_factor = {1, 1, 1};

  split_shape(split_factor, sh, 3, 1000);  // minimum volume is equal to the input volume, no split
  ASSERT_EQ(split_factor[0], 1);
  ASSERT_EQ(split_factor[1], 1);
  ASSERT_EQ(split_factor[2], 1);

  split_shape(split_factor, sh, 10000, 1);  // requested more than possible
  ASSERT_EQ(split_factor[0], 10);
  ASSERT_EQ(split_factor[1], 10);
  ASSERT_EQ(split_factor[2], 10);

  split_shape(split_factor, sh, 10, 10);  // split across first dimension only
  ASSERT_EQ(split_factor[0], 10);
  ASSERT_EQ(split_factor[1], 1);
  ASSERT_EQ(split_factor[2], 1);

  split_shape(split_factor, sh, 3, 10);  // split across first dimension only
  ASSERT_EQ(split_factor[0], 10);  // Due to 3 * 4 > 10
  ASSERT_EQ(split_factor[1], 1);
  ASSERT_EQ(split_factor[2], 1);

  split_shape(split_factor, sh, 2, 10);  // split across first dimension only
  ASSERT_EQ(split_factor[0], 2);
  ASSERT_EQ(split_factor[1], 1);
  ASSERT_EQ(split_factor[2], 1);

  split_shape(split_factor, sh, 20, 10);  // split across first and second dimensions
  ASSERT_EQ(split_factor[0], 10);
  ASSERT_EQ(split_factor[1], 2);
  ASSERT_EQ(split_factor[2], 1);

  split_shape(split_factor, sh, 16, 1);  // split across first and second dimensions
  ASSERT_EQ(split_factor[0], 10);
  ASSERT_EQ(split_factor[1], 2);
  ASSERT_EQ(split_factor[2], 1);

  split_shape(split_factor, sh, 19, 1);  // split across first and second dimensions
  ASSERT_EQ(split_factor[0], 10);
  ASSERT_EQ(split_factor[1], 2);  // due to 2 * 4 < 10
  ASSERT_EQ(split_factor[2], 1);

  split_shape(split_factor, sh, 24, 1);  // split across first and second dimensions
  ASSERT_EQ(split_factor[0], 10);
  ASSERT_EQ(split_factor[1], 10);  // due to 3 * 4 > 10
  ASSERT_EQ(split_factor[2], 1);

  split_shape(split_factor, sh, 1000, 10);  // split until the volume is small enough
  ASSERT_EQ(split_factor[0], 10);
  ASSERT_EQ(split_factor[1], 10);
  ASSERT_EQ(split_factor[2], 1);
}

}  // namespace kernels
}  // namespace dali
