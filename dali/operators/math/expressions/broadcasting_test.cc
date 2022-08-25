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
    TensorShape<> result0, result1;
    ASSERT_TRUE(CanBroadcast(a, b));
    ASSERT_TRUE(CanBroadcast(b, a));
    BroadcastShape(result0, a, b);
    EXPECT_EQ(expected, result0);
    BroadcastShape(result1, a, b);
    EXPECT_EQ(expected, result1);
  };
  test({1, 3, 2}, {3, 2}, {1, 3, 1});
  test({10, 10, 2}, {10, 1, 2}, {1, 10, 2});
  test({10, 10, 2}, {10, 10, 2}, {});
  test({10, 10, 3}, {10, 10, 1}, {3});
  test({}, {}, {});
  test({1}, {}, {1});

  auto test_fail = [](TensorShape<> a, TensorShape<> b) {
    TensorShape<> result;
    ASSERT_FALSE(CanBroadcast(a, b));
    ASSERT_FALSE(CanBroadcast(b, a));
    ASSERT_THROW(BroadcastShape(result, a, b), std::runtime_error);
    ASSERT_THROW(BroadcastShape(result, b, a), std::runtime_error);
  };
  test_fail({3, 2}, {2, 3});
  test_fail({2}, {2, 3});
}

TEST(ArithmeticOpsBroadcastingTest, BroadcastTensorListShape) {
  auto test = [](TensorListShape<> expected, TensorListShape<> a, TensorListShape<> b) {
    TensorListShape result0, result1;
    ASSERT_TRUE(CanBroadcast(a, b));
    ASSERT_TRUE(CanBroadcast(b, a));
    BroadcastShape(result0, a, b);
    EXPECT_EQ(expected, result0);
    BroadcastShape(result1, a, b);
    EXPECT_EQ(expected, result1);
  };
  test({{2, 2, 3}}, {{2, 2, 3}}, {{2, 1, 3}});
  test({{1, 3, 2}, {3, 2, 4}, {4, 2, 1}},
       {{1, 3, 2}, {3, 2, 1}, {4, 1, 1}}, {{1, 1, 2}, {3, 2, 4}, {1, 2, 1}});
  test({{1, 3, 2}, {3, 2, 4}, {2, 4, 1}},
       {{1, 3, 2}, {3, 2, 1}, {2, 1, 1}}, {{1, 1, 2}, {3, 2, 4}, {1, 4, 1}});
  test({{1, 3, 2}, {3, 2, 4}, {2, 4, 1}},
       {{3, 1}, {2, 1}, {1, 1}}, {{1, 1, 2}, {3, 1, 4}, {2, 4, 1}});

  auto test_fail = [](TensorListShape<> a, TensorListShape<> b) {
    TensorListShape<> result;
    ASSERT_FALSE(CanBroadcast(a, b));
    ASSERT_FALSE(CanBroadcast(b, a));
    ASSERT_THROW(BroadcastShape(result, a, b), std::runtime_error);
    ASSERT_THROW(BroadcastShape(result, b, a), std::runtime_error);
  };
  test_fail({{1, 3, 2}, {3, 2, 1}, {1, 1, 1}}, {{1, 1, 2}, {3, 3, 4}, {4, 1, 1}});
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


TEST(ArithmeticOpsBroadcastingTest, BroadcastSpanShapes) {
  auto test = [](TensorShape<> expected, auto&&... shapes) {
    TensorShape result0;
    ASSERT_TRUE(CanBroadcast(shapes...));
    BroadcastShape(result0, shapes...);
    EXPECT_EQ(expected, result0);
  };

  test({1, 2, 2, 3, 2},
       TensorShape<>{3, 2}, TensorShape<>{1, 3, 1}, TensorShape<>{1, 2, 2, 1, 2});
  test({1, 3, 20},
       TensorShape<>{3, 20}, TensorShape<>{1, 3, 1}, TensorShape<>{20});
  test({1, 3, 20},
       TensorShape<>{3, 20}, TensorShape<>{1, 3, 1}, TensorShape<>{});

  auto test_fail = [](auto&&... shapes) {
    TensorShape<> result;
    ASSERT_FALSE(CanBroadcast(shapes...));
    ASSERT_THROW(BroadcastShape(result, shapes...), std::runtime_error);
  };
  test_fail(TensorShape<>{3, 2}, TensorShape<>{1, 3, 3}, TensorShape<>{1, 1, 1, 3, 3});
}


TEST(ArithmeticOpsBroadcastingTest, NeedBroadcastShape) {
  auto test_no_need = [](TensorShape<> a, TensorShape<> b) {
    EXPECT_FALSE(NeedBroadcasting(a, b));
    EXPECT_FALSE(NeedBroadcasting(b, a));
  };
  auto test_need = [](TensorShape<> a, TensorShape<> b) {
    EXPECT_TRUE(NeedBroadcasting(a, b));
    EXPECT_TRUE(NeedBroadcasting(b, a));
  };

  test_no_need({1, 2, 3}, {1, 2, 3});
  test_no_need({1, 2, 3}, {});   // scalar
  test_no_need({1, 2, 3}, {1});  // scalar-like

  test_need({1, 2, 3}, {1, 1, 3});
  test_need({1, 2, 2, 3}, {2, 1});
}

TEST(ArithmeticOpsBroadcastingTest, NeedBroadcastTensorListShape) {
  auto test_no_need = [](TensorListShape<> a, TensorListShape<> b) {
    EXPECT_FALSE(NeedBroadcasting(a, b));
    EXPECT_FALSE(NeedBroadcasting(b, a));
  };
  auto test_need = [](TensorListShape<> a, TensorListShape<> b) {
    EXPECT_TRUE(NeedBroadcasting(a, b));
    EXPECT_TRUE(NeedBroadcasting(b, a));
  };

  test_no_need({{1, 2, 3}, {1, 2, 1}}, {{1, 2, 3}, {1, 2, 1}});
  test_no_need({{2}, {2}}, {{1}, {1}});  // scalar-like

  test_need({{1, 2, 3}, {1, 2, 1}}, {{1, 2, 1}, {1, 2, 3}});
}


}  // namespace test
}  // namespace dali
