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
#include "dali/operators/math/expressions/broadcasting.h"
#include "dali/core/tensor_shape.h"
#include "dali/kernels/common/utils.h"

namespace dali {
namespace test {

TEST(ArithmeticOpsBroadcastingTest, BroadcastShape) {
  auto test = [](TensorShape<> expected, TensorShape<> a, TensorShape<> b) {
    EXPECT_EQ(expected, BroadcastShape(a, b));
    EXPECT_EQ(expected, BroadcastShape(b, a));
  };
  test({1, 3, 2}, {3, 2}, {1, 3, 1});
  test({10, 10, 2}, {10, 1, 2}, {1, 10, 2});
  test({10, 10, 2}, {10, 10, 2}, {});
  test({10, 10, 3}, {10, 10, 1}, {3});
  test({}, {}, {});
  test({1}, {}, {1});

  auto test_fail = [](TensorShape<> a, TensorShape<> b) {
    ASSERT_THROW(BroadcastShape(a, b), std::runtime_error);
  };
  test_fail({3, 2}, {2, 3});
  test_fail({2}, {2, 3});
}

TEST(ArithmeticOpsBroadcastingTest, StridesForBroadcasting) {
  TensorShape<> orig_shape = {10, 1, 2};
  TensorShape<> orig_strides;
  kernels::CalcStrides(orig_strides, orig_shape);

  TensorShape<> out_sh = {10, 10, 2};
  auto strides = StridesForBroadcasting(out_sh, orig_shape, orig_strides);
  EXPECT_EQ(orig_strides[0], strides[0]);
  EXPECT_EQ(0, strides[1]);
  EXPECT_EQ(orig_strides[2], strides[2]);
}

TEST(ArithmeticOpsBroadcastingTest, SimplifyShapesForBroadcasting) {
  // Only collapsing dims that are not broadcasted
  {
    TensorShape<> a = {10, 2, 2, 3};
    TensorShape<> b = {10, 2, 1, 3};
    SimplifyShapesForBroadcasting(a, b);
    TensorShape<> simple_a = {20, 2, 3};
    TensorShape<> simple_b = {20, 1, 3};
    EXPECT_EQ(simple_a, a);
    EXPECT_EQ(simple_b, b);
  }

  // First expanding channels, then collapsing
  {
    TensorShape<> a = {2, 1, 10, 2, 1, 3};
    TensorShape<> b =       {10, 2, 2, 3};
    SimplifyShapesForBroadcasting(a, b);
    TensorShape<> simple_a = {2, 20, 1, 3};
    TensorShape<> simple_b = {1, 20, 2, 3};
    EXPECT_EQ(simple_a, a);
    EXPECT_EQ(simple_b, b);
  }

  // No broadcasting, collapsing to one dim
  {
    TensorShape<> a = {2, 1, 10, 2, 1, 3};
    TensorShape<> b = {2, 1, 10, 2, 1, 3};
    SimplifyShapesForBroadcasting(a, b);
    TensorShape<> simple_a = {120};
    TensorShape<> simple_b = {120};
    EXPECT_EQ(simple_a, a);
    EXPECT_EQ(simple_b, b);
  }
}

}  // namespace test
}  // namespace dali
