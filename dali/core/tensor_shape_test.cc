// Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_view.h"
#include "dali/core/tensor_shape_print.h"

namespace dali {
namespace kernels {

static_assert(compile_time_size_impl<int[41]>::value == 41, "Unexpected value");
static_assert(compile_time_size_impl<std::array<uint16_t, 123>>::value == 123,
  "Unexpected array size");
static_assert(compile_time_size_impl<TensorShape<9>>::value == 9,
  "Unexpected tensor shape size");
static_assert(compile_time_size_impl<TensorShape<>>::value == DynamicDimensions,
  "Unexpected tensor shape size");

TEST(TensorShapeTest, DefaultStaticShapeConstructor) {
  TensorShape<0> zero_tensor;
  ASSERT_EQ(zero_tensor.size(), 0);

  constexpr int test_dim = 5;
  // Default constructor
  TensorShape<test_dim> empty_tensor;
  for (int i = 0; i < test_dim; i++) {
    EXPECT_EQ(empty_tensor[i], int64_t(0));
  }
}

TEST(TensorShapeTest, StaticShapeConstructor) {
  // std::array and expanded list constructor
  constexpr int test_dim = 5;
  std::array<int64_t, test_dim> test_shape = {1, 2, 3, 4, 5};
  TensorShape<test_dim> a(test_shape);
  TensorShape<test_dim> b(test_shape[0], test_shape[1], test_shape[2], test_shape[3],
                          test_shape[4]);
  ASSERT_EQ(a.size(), test_dim);
  ASSERT_EQ(b.size(), test_dim);
  for (int i = 0; i < test_dim; i++) {
    EXPECT_EQ(a[i], test_shape[i]);
    EXPECT_EQ(b[i], test_shape[i]);
  }
}

TEST(TensorShapeTest, CopyAssignStaticShapeConstructor) {
  // Copy constructor
  constexpr int test_dim = 5;
  std::array<int64_t, test_dim> test_shape = {1, 2, 3, 4, 5};
  TensorShape<test_dim> a(test_shape);
  TensorShape<test_dim> check_construct(a);
  for (int i = 0; i < test_dim; i++) {
    EXPECT_EQ(check_construct[i], a[i]);
  }

  // Assignement
  TensorShape<test_dim> check_assign;
  check_assign = a;
  for (int i = 0; i < test_dim; i++) {
    EXPECT_EQ(check_assign[i], a[i]);
  }
}

TEST(TensorShapeTest, DefaultDynamicShapeConstructor) {
  // Default
  TensorShape<DynamicDimensions> zero_tensor;
  ASSERT_EQ(zero_tensor.size(), 0);
}

TEST(TensorShapeTest, DynamicShapeIteratorRangeConstructor) {
  std::vector<int64_t> test_shape = {1, 2, 3, 4, 5};
  TensorShape<DynamicDimensions> a(test_shape[0], test_shape[1], test_shape[2], test_shape[3],
                                   test_shape[4]);
  TensorShape<DynamicDimensions> b(test_shape.begin(), test_shape.end());
  ASSERT_EQ(a.size(), test_shape.size());
  ASSERT_EQ(b.size(), test_shape.size());
  for (size_t i = 0; i < test_shape.size(); i++) {
    EXPECT_EQ(a[i], test_shape[i]);
    EXPECT_EQ(b[i], test_shape[i]);
  }
}

TEST(TensorShapeTest, FromArrayDynamicShapeConstructor) {
  constexpr int test_dim = 5;
  std::array<int64_t, test_dim> test_shape_arr = {1, 2, 3, 4, 5};
  TensorShape<DynamicDimensions> a(test_shape_arr);
  ASSERT_EQ(a.size(), test_dim);
  for (int i = 0; i < test_dim; i++) {
    EXPECT_EQ(a[i], test_shape_arr[i]);
  }
}

TEST(TensorShapeTest, FromVectorDynamicShapeConstructor) {
  std::vector<int64_t> test_shape_vec = {1, 2, 3, 4, 5, 6, 7};
  TensorShape<DynamicDimensions> b(test_shape_vec);
  ASSERT_EQ(b.size(), test_shape_vec.size());
  for (size_t i = 0; i < test_shape_vec.size(); i++) {
    ASSERT_EQ(b[i], test_shape_vec[i]);
  }
}

TEST(TensorShapeTest, ExpandedDynamicShapeConstructor) {
  // Expanded arguments constructor
  TensorShape<DynamicDimensions> c(1);
  ASSERT_EQ(c.size(), 1);
  ASSERT_EQ(c[0], 1);

  TensorShape<DynamicDimensions> d(1, 2, 3, 4);
  ASSERT_EQ(d.size(), 4);
  EXPECT_EQ(d[0], 1);
  EXPECT_EQ(d[1], 2);
  EXPECT_EQ(d[2], 3);
  EXPECT_EQ(d[3], 4);
}

TEST(TensorShapeTest, CopyAssignDynamicShapeConstructor) {
  constexpr int test_dim = 5;
  std::array<int64_t, test_dim> test_shape_arr = {1, 2, 3, 4, 5};
  TensorShape<DynamicDimensions> a(test_shape_arr);
  // Copy constructor
  TensorShape<DynamicDimensions> check_construct(a);
  for (int i = 0; i < test_dim; i++) {
    EXPECT_EQ(check_construct[i], a[i]);
  }

  // Asignement
  TensorShape<DynamicDimensions> check_assign;
  check_assign = a;
  ASSERT_EQ(check_assign.size(), a.size());
  for (int i = 0; i < a.size(); i++) {
    EXPECT_EQ(check_assign[i], a[i]);
  }

  // Second asignement to the same value
  std::vector<int64_t> test_shape_vec = {1, 2, 3, 4, 5, 6, 7};
  TensorShape<DynamicDimensions> b(test_shape_vec);
  check_assign = b;
  ASSERT_EQ(check_assign.size(), b.size());
  for (int i = 0; i < b.size(); i++) {
    EXPECT_EQ(check_assign[i], b[i]);
  }
}

TEST(TensorShapeTest, MoveDynamicShapeConstructor) {
  constexpr int test_dim = 5;
  std::array<int64_t, test_dim> test_shape_arr = {1, 2, 3, 4, 5};
  TensorShape<DynamicDimensions> tmp{test_shape_arr};
  // Move rvalue
  TensorShape<DynamicDimensions> check_move_construct(std::move(tmp));
  ASSERT_TRUE(tmp.shape.empty());
  ASSERT_EQ(check_move_construct.size(), test_dim);
  for (int i = 0; i < test_dim; i++) {
    EXPECT_EQ(check_move_construct[i], test_shape_arr[i]);
  }

  // Assignement for rvalue
  TensorShape<DynamicDimensions> check_move_assign;
  TensorShape<DynamicDimensions> tmp2{test_shape_arr};
  check_move_assign = std::move(tmp2);
  ASSERT_TRUE(tmp2.shape.empty());
  ASSERT_EQ(check_move_assign.size(), test_dim);
  for (int i = 0; i < test_dim; i++) {
    EXPECT_EQ(check_move_assign[i], test_shape_arr[i]);
  }
}

TEST(TensorShapeTest, StaticDynamicConversions) {
  TensorShape<3> static_shape_3(1, 2, 3);
  TensorShape<5> static_shape_5(1, 2, 3, 4, 5);

  TensorShape<DynamicDimensions> check_construct_3(static_shape_3);
  ASSERT_EQ(check_construct_3.size(), static_shape_3.size());
  for (int i = 0; i < static_shape_3.size(); i++) {
    EXPECT_EQ(check_construct_3[i], static_shape_3[i]);
  }

  TensorShape<DynamicDimensions> check_construct_5(static_shape_5);
  ASSERT_EQ(check_construct_5.size(), static_shape_5.size());
  for (int i = 0; i < static_shape_5.size(); i++) {
    EXPECT_EQ(check_construct_5[i], static_shape_5[i]);
  }

  TensorShape<DynamicDimensions> check_assign;
  check_assign = static_shape_3;
  ASSERT_EQ(check_assign.size(), static_shape_3.size());
  for (int i = 0; i < static_shape_3.size(); i++) {
    EXPECT_EQ(check_assign[i], static_shape_3[i]);
  }

  check_assign = static_shape_5;
  ASSERT_EQ(check_assign.size(), static_shape_5.size());
  for (int i = 0; i < static_shape_5.size(); i++) {
    EXPECT_EQ(check_assign[i], static_shape_5[i]);
  }

  auto s3 = TensorShape<3>{2, 4, 6};
  check_assign = s3;
  static_shape_3 = check_assign.to_static<3>();
  for (int i = 0; i < s3.size(); i++) {
    EXPECT_EQ(static_shape_3[i], s3[i]);
  }
}

TEST(TensorListShapeTest, UniformListShapeDynamic) {
  auto dynamic0 = uniform_list_shape(0, std::vector<int64_t>{});
  EXPECT_EQ(dynamic0.size(), 0);
  EXPECT_EQ(dynamic0.sample_dim(), 0);
  EXPECT_EQ(dynamic0.shapes.size(), 0);
  auto dynamic1 = uniform_list_shape(0, std::vector<int64_t>{0, 1, 2, 3});
  EXPECT_EQ(dynamic1.size(), 0);
  EXPECT_EQ(dynamic1.sample_dim(), 4);
  EXPECT_EQ(dynamic1.shapes.size(), 0);
  auto dynamic2 = uniform_list_shape(2, std::vector<int64_t>{});
  EXPECT_EQ(dynamic2.size(), 2);
  EXPECT_EQ(dynamic2.sample_dim(), 0);
  EXPECT_EQ(dynamic2.shapes.size(), 0);
  auto dynamic3 = uniform_list_shape(2, std::vector<int64_t>{0, 1, 2, 3});
  EXPECT_EQ(dynamic3.size(), 2);
  EXPECT_EQ(dynamic3.sample_dim(), 4);
  EXPECT_EQ(dynamic3.shapes.size(), 8);
}


TEST(TensorListShapeTest, UniformListShapeStatic) {
  auto static0 = uniform_list_shape<0>(0, std::vector<int64_t>{});
  EXPECT_EQ(static0.size(), 0);
  EXPECT_EQ(static0.sample_dim(), 0);
  EXPECT_EQ(static0.shapes.size(), 0);
  auto static1 = uniform_list_shape<4>(0, std::vector<int64_t>{0, 1, 2, 3});
  EXPECT_EQ(static1.size(), 0);
  EXPECT_EQ(static1.sample_dim(), 4);
  EXPECT_EQ(static1.shapes.size(), 0);
  auto static2 = uniform_list_shape<0>(2, std::vector<int64_t>{});
  EXPECT_EQ(static2.size(), 2);
  EXPECT_EQ(static2.sample_dim(), 0);
  EXPECT_EQ(static2.shapes.size(), 0);
  auto static3 = uniform_list_shape<4>(2, std::vector<int64_t>{0, 1, 2, 3});
  EXPECT_EQ(static3.size(), 2);
  EXPECT_EQ(static3.sample_dim(), 4);
  EXPECT_EQ(static3.shapes.size(), 8);
}

TEST(TensorShapeTest, StaticComparisons) {
  // Static ndim
  EXPECT_TRUE(TensorShape<1>(1) == TensorShape<1>(1));
  EXPECT_FALSE(TensorShape<1>(1) != TensorShape<1>(1));

  EXPECT_FALSE(TensorShape<1>(1) == TensorShape<1>(2));
  EXPECT_TRUE(TensorShape<1>(1) != TensorShape<1>(2));

  EXPECT_TRUE(TensorShape<3>(1, 2, 3) == TensorShape<3>(1, 2, 3));
  EXPECT_FALSE(TensorShape<3>(1, 2, 3) != TensorShape<3>(1, 2, 3));

  EXPECT_FALSE(TensorShape<3>(1, 2, 3) == TensorShape<3>(1, 4, 3));
  EXPECT_TRUE(TensorShape<3>(1, 2, 3) != TensorShape<3>(1, 4, 3));

  EXPECT_FALSE(TensorShape<1>(1) == TensorShape<2>(1, 2));
  EXPECT_TRUE(TensorShape<1>(1) != TensorShape<2>(1, 2));
  EXPECT_FALSE(TensorShape<2>(1, 2) == TensorShape<1>(1));
  EXPECT_TRUE(TensorShape<2>(1, 2) != TensorShape<1>(1));
}

TEST(TensorShapeTest, DynamicComparisons) {
  // Dynamic ndim
  EXPECT_TRUE(TensorShape<DynamicDimensions>(1) == TensorShape<DynamicDimensions>(1));
  EXPECT_FALSE(TensorShape<DynamicDimensions>(1) != TensorShape<DynamicDimensions>(1));

  EXPECT_FALSE(TensorShape<DynamicDimensions>(1) == TensorShape<DynamicDimensions>(2));
  EXPECT_TRUE(TensorShape<DynamicDimensions>(1) != TensorShape<DynamicDimensions>(2));

  EXPECT_TRUE(TensorShape<DynamicDimensions>(1, 2, 3) == TensorShape<DynamicDimensions>(1, 2, 3));
  EXPECT_FALSE(TensorShape<DynamicDimensions>(1, 2, 3) != TensorShape<DynamicDimensions>(1, 2, 3));

  EXPECT_FALSE(TensorShape<DynamicDimensions>(1, 2, 3) == TensorShape<DynamicDimensions>(1, 4, 3));
  EXPECT_TRUE(TensorShape<DynamicDimensions>(1, 2, 3) != TensorShape<DynamicDimensions>(1, 4, 3));

  EXPECT_FALSE(TensorShape<DynamicDimensions>(1) == TensorShape<DynamicDimensions>(1, 2));
  EXPECT_TRUE(TensorShape<DynamicDimensions>(1) != TensorShape<DynamicDimensions>(1, 2));
  EXPECT_FALSE(TensorShape<DynamicDimensions>(1, 2) == TensorShape<DynamicDimensions>(1));
  EXPECT_TRUE(TensorShape<DynamicDimensions>(1, 2) != TensorShape<DynamicDimensions>(1));
}

TEST(TensorShapeTest, MixedComparisons) {
  // Mixed ndim
  EXPECT_TRUE(TensorShape<1>(1) == TensorShape<DynamicDimensions>(1));
  EXPECT_FALSE(TensorShape<1>(1) != TensorShape<DynamicDimensions>(1));
  EXPECT_TRUE(TensorShape<DynamicDimensions>(1) == TensorShape<1>(1));
  EXPECT_FALSE(TensorShape<DynamicDimensions>(1) != TensorShape<1>(1));

  EXPECT_FALSE(TensorShape<1>(1) == TensorShape<DynamicDimensions>(2));
  EXPECT_TRUE(TensorShape<1>(1) != TensorShape<DynamicDimensions>(2));
  EXPECT_FALSE(TensorShape<DynamicDimensions>(1) == TensorShape<1>(2));
  EXPECT_TRUE(TensorShape<DynamicDimensions>(1) != TensorShape<1>(2));

  EXPECT_TRUE(TensorShape<3>(1, 2, 3) == TensorShape<DynamicDimensions>(1, 2, 3));
  EXPECT_FALSE(TensorShape<3>(1, 2, 3) != TensorShape<DynamicDimensions>(1, 2, 3));
  EXPECT_TRUE(TensorShape<DynamicDimensions>(1, 2, 3) == TensorShape<3>(1, 2, 3));
  EXPECT_FALSE(TensorShape<DynamicDimensions>(1, 2, 3) != TensorShape<3>(1, 2, 3));

  EXPECT_FALSE(TensorShape<3>(1, 2, 3) == TensorShape<DynamicDimensions>(1, 4, 3));
  EXPECT_TRUE(TensorShape<3>(1, 2, 3) != TensorShape<DynamicDimensions>(1, 4, 3));
  EXPECT_FALSE(TensorShape<DynamicDimensions>(1, 2, 3) == TensorShape<3>(1, 4, 3));
  EXPECT_TRUE(TensorShape<DynamicDimensions>(1, 2, 3) != TensorShape<3>(1, 4, 3));

  EXPECT_FALSE(TensorShape<1>(1) == TensorShape<DynamicDimensions>(1, 2));
  EXPECT_TRUE(TensorShape<1>(1) != TensorShape<DynamicDimensions>(1, 2));
  EXPECT_FALSE(TensorShape<2>(1, 2) == TensorShape<DynamicDimensions>(1));
  EXPECT_TRUE(TensorShape<2>(1, 2) != TensorShape<DynamicDimensions>(1));
  EXPECT_FALSE(TensorShape<DynamicDimensions>(1) == TensorShape<2>(1, 2));
  EXPECT_TRUE(TensorShape<DynamicDimensions>(1) != TensorShape<2>(1, 2));
  EXPECT_FALSE(TensorShape<DynamicDimensions>(1, 2) == TensorShape<1>(1));
  EXPECT_TRUE(TensorShape<DynamicDimensions>(1, 2) != TensorShape<1>(1));
}

TEST(TensorShapeTest, RangeLoop) {
  TensorShape<5> ts{0, 1, 2, 3, 4};
  int expected = 0;
  for (auto s : ts) {
    EXPECT_EQ(s, expected);
    expected++;
  }
}

TEST(TensorShapeTest, FirstStaticOnStatic) {
  TensorShape<5> ts(1, 2, 3, 4, 5);
  EXPECT_EQ(ts.first<0>(), TensorShape<0>());
  EXPECT_EQ(ts.first<1>(), TensorShape<1>(1));
  EXPECT_EQ(ts.first<2>(), TensorShape<2>(1, 2));
  EXPECT_EQ(ts.first<3>(), TensorShape<3>(1, 2, 3));
  EXPECT_EQ(ts.first<4>(), TensorShape<4>(1, 2, 3, 4));
  EXPECT_EQ(ts.first<5>(), TensorShape<5>(1, 2, 3, 4, 5));
}

TEST(TensorShapeTest, LastStaticOnStatic) {
  TensorShape<5> ts(1, 2, 3, 4, 5);
  EXPECT_EQ(ts.last<0>(), TensorShape<0>());
  EXPECT_EQ(ts.last<1>(), TensorShape<1>(5));
  EXPECT_EQ(ts.last<2>(), TensorShape<2>(4, 5));
  EXPECT_EQ(ts.last<3>(), TensorShape<3>(3, 4, 5));
  EXPECT_EQ(ts.last<4>(), TensorShape<4>(2, 3, 4, 5));
  EXPECT_EQ(ts.last<5>(), TensorShape<5>(1, 2, 3, 4, 5));
}

TEST(TensorShapeTest, FirstStaticOnDynamic) {
  TensorShape<DynamicDimensions> ts(1, 2, 3, 4, 5);
  EXPECT_EQ(ts.first<0>(), TensorShape<DynamicDimensions>());
  EXPECT_EQ(ts.first<1>(), TensorShape<DynamicDimensions>(1));
  EXPECT_EQ(ts.first<2>(), TensorShape<DynamicDimensions>(1, 2));
  EXPECT_EQ(ts.first<3>(), TensorShape<DynamicDimensions>(1, 2, 3));
  EXPECT_EQ(ts.first<4>(), TensorShape<DynamicDimensions>(1, 2, 3, 4));
  EXPECT_EQ(ts.first<5>(), TensorShape<DynamicDimensions>(1, 2, 3, 4, 5));
}

TEST(TensorShapeTest, LastStaticOnDynamic) {
  TensorShape<DynamicDimensions> ts(1, 2, 3, 4, 5);
  EXPECT_EQ(ts.last<0>(), TensorShape<DynamicDimensions>());
  EXPECT_EQ(ts.last<1>(), TensorShape<DynamicDimensions>(5));
  EXPECT_EQ(ts.last<2>(), TensorShape<DynamicDimensions>(4, 5));
  EXPECT_EQ(ts.last<3>(), TensorShape<DynamicDimensions>(3, 4, 5));
  EXPECT_EQ(ts.last<4>(), TensorShape<DynamicDimensions>(2, 3, 4, 5));
  EXPECT_EQ(ts.last<5>(), TensorShape<DynamicDimensions>(1, 2, 3, 4, 5));
}

TEST(TensorShapeTest, FirstDynamicOnStatic) {
  TensorShape<5> ts(1, 2, 3, 4, 5);
  EXPECT_EQ(ts.first(0), TensorShape<0>());
  EXPECT_EQ(ts.first(1), TensorShape<1>(1));
  EXPECT_EQ(ts.first(2), TensorShape<2>(1, 2));
  EXPECT_EQ(ts.first(3), TensorShape<3>(1, 2, 3));
  EXPECT_EQ(ts.first(4), TensorShape<4>(1, 2, 3, 4));
  EXPECT_EQ(ts.first(5), TensorShape<5>(1, 2, 3, 4, 5));
}

TEST(TensorShapeTest, LastDynamicOnStatic) {
  TensorShape<5> ts(1, 2, 3, 4, 5);
  EXPECT_EQ(ts.last(0), TensorShape<0>());
  EXPECT_EQ(ts.last(1), TensorShape<1>(5));
  EXPECT_EQ(ts.last(2), TensorShape<2>(4, 5));
  EXPECT_EQ(ts.last(3), TensorShape<3>(3, 4, 5));
  EXPECT_EQ(ts.last(4), TensorShape<4>(2, 3, 4, 5));
  EXPECT_EQ(ts.last(5), TensorShape<5>(1, 2, 3, 4, 5));
}

TEST(TensorShapeTest, FirstDynamicOnDynamic) {
  TensorShape<DynamicDimensions> ts(1, 2, 3, 4, 5);
  EXPECT_EQ(ts.first(0), TensorShape<DynamicDimensions>());
  EXPECT_EQ(ts.first(1), TensorShape<DynamicDimensions>(1));
  EXPECT_EQ(ts.first(2), TensorShape<DynamicDimensions>(1, 2));
  EXPECT_EQ(ts.first(3), TensorShape<DynamicDimensions>(1, 2, 3));
  EXPECT_EQ(ts.first(4), TensorShape<DynamicDimensions>(1, 2, 3, 4));
  EXPECT_EQ(ts.first(5), TensorShape<DynamicDimensions>(1, 2, 3, 4, 5));
}

TEST(TensorShapeTest, LastDynamicOnDynamic) {
  TensorShape<DynamicDimensions> ts(1, 2, 3, 4, 5);
  EXPECT_EQ(ts.last(0), TensorShape<DynamicDimensions>());
  EXPECT_EQ(ts.last(1), TensorShape<DynamicDimensions>(5));
  EXPECT_EQ(ts.last(2), TensorShape<DynamicDimensions>(4, 5));
  EXPECT_EQ(ts.last(3), TensorShape<DynamicDimensions>(3, 4, 5));
  EXPECT_EQ(ts.last(4), TensorShape<DynamicDimensions>(2, 3, 4, 5));
  EXPECT_EQ(ts.last(5), TensorShape<DynamicDimensions>(1, 2, 3, 4, 5));
}

TEST(TensorShapeTest, ConcatenationStaticIdentityElem) {
  EXPECT_EQ(shape_cat(TensorShape<0>(), TensorShape<0>()), TensorShape<0>());
  EXPECT_EQ(shape_cat(TensorShape<1>(1), TensorShape<0>()), TensorShape<1>(1));
  EXPECT_EQ(shape_cat(TensorShape<0>(), TensorShape<1>(1)), TensorShape<1>(1));
}

TEST(TensorShapeTest, ConcatenationStatic) {
  EXPECT_EQ(shape_cat(TensorShape<2>(1, 2), TensorShape<3>(1, 2, 3)),
            TensorShape<5>(1, 2, 1, 2, 3));
}

TEST(TensorShapeTest, ConcatenationDynamicIdentityElem) {
  EXPECT_EQ(shape_cat(TensorShape<DynamicDimensions>(), TensorShape<DynamicDimensions>()),
            TensorShape<DynamicDimensions>());
  EXPECT_EQ(shape_cat(TensorShape<DynamicDimensions>(1), TensorShape<DynamicDimensions>()),
            TensorShape<DynamicDimensions>(1));
  EXPECT_EQ(shape_cat(TensorShape<DynamicDimensions>(), TensorShape<DynamicDimensions>(1)),
            TensorShape<DynamicDimensions>(1));
}

TEST(TensorShapeTest, ConcatenationDynamic) {
  EXPECT_EQ(
      shape_cat(TensorShape<DynamicDimensions>(1, 2), TensorShape<DynamicDimensions>(1, 2, 3)),
      TensorShape<DynamicDimensions>(1, 2, 1, 2, 3));
}

TEST(TensorShapeTest, ConcatenationMixedIdentityElem) {
  EXPECT_EQ(shape_cat(TensorShape<DynamicDimensions>(), TensorShape<0>()),
            TensorShape<DynamicDimensions>());
  EXPECT_EQ(shape_cat(TensorShape<DynamicDimensions>(1), TensorShape<0>()),
            TensorShape<DynamicDimensions>(1));
  EXPECT_EQ(shape_cat(TensorShape<DynamicDimensions>(), TensorShape<1>(1)),
            TensorShape<DynamicDimensions>(1));
  EXPECT_EQ(shape_cat(TensorShape<0>(), TensorShape<DynamicDimensions>()),
            TensorShape<DynamicDimensions>());
  EXPECT_EQ(shape_cat(TensorShape<1>(1), TensorShape<DynamicDimensions>()),
            TensorShape<DynamicDimensions>(1));
  EXPECT_EQ(shape_cat(TensorShape<0>(), TensorShape<DynamicDimensions>(1)),
            TensorShape<DynamicDimensions>(1));
}

TEST(TensorShapeTest, ConcatenationMixed) {
  EXPECT_EQ(shape_cat(TensorShape<DynamicDimensions>(1, 2), TensorShape<3>(1, 2, 3)),
            TensorShape<DynamicDimensions>(1, 2, 1, 2, 3));

  EXPECT_EQ(shape_cat(TensorShape<2>(1, 2), TensorShape<DynamicDimensions>(1, 2, 3)),
            TensorShape<DynamicDimensions>(1, 2, 1, 2, 3));
}

TEST(TensorShapeTest, ConstructorFromIntegerCollection) {
  TensorShape<> ref{1, 2, 3, 4, 5};
  std::vector<int> vec_i{1, 2, 3, 4, 5};

  EXPECT_EQ(ref, TensorShape<>(vec_i));
  EXPECT_EQ(ref, TensorShape<>(vec_i.begin(), vec_i.end()));
  TensorShape<> sh1;
  sh1 = vec_i;
  EXPECT_EQ(ref, sh1);

  EXPECT_EQ(ref, TensorShape<5>(vec_i));
  EXPECT_EQ(ref, TensorShape<5>(vec_i.begin(), vec_i.end()));
  TensorShape<5> sh2;
  sh2 = vec_i;
  EXPECT_EQ(ref, sh2);
}

TEST(VolumeTest, Result) {
  EXPECT_EQ(volume(std::vector<int64_t>{}), 1);
  EXPECT_EQ(volume(std::vector<int64_t>{2}), 2);
  EXPECT_EQ(volume(std::vector<int64_t>{1, 2}), 2);
  EXPECT_EQ(volume(std::vector<int64_t>{1, 2, 3, 4, 5}), 1 * 2 * 3 * 4 * 5);
}

TEST(FlattenTest, StaticTensorShape) {
  auto shapes_vec = std::vector<TensorShape<3>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  auto expected = std::vector<int64_t>{1, 2, 3, 2, 3, 4, 3, 4, 5};
  EXPECT_EQ(flatten_shapes(shapes_vec), expected);
}

TEST(FlattenTest, DynamicTensorShape) {
  auto shapes_vec = std::vector<TensorShape<>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  auto vec_vec = std::vector<std::vector<int64_t>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  auto expected = std::vector<int64_t>{1, 2, 3, 2, 3, 4, 3, 4, 5};
  EXPECT_EQ(flatten_shapes(shapes_vec), expected);
  EXPECT_EQ(flatten_shapes(vec_vec), expected);
}

TEST(CalculatePointersTest, Result) {
  TensorListShape<3> tls_static({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});
  TensorListShape<> tls_dynamic(
      std::vector<TensorShape<DynamicDimensions>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});
  static const char ptr[100] = {};
  auto static_ptrs = calculate_pointers(ptr, tls_static);
  auto dynamic_ptrs = calculate_pointers(ptr, tls_dynamic);
  auto expected = std::vector<const char *>{ptr + 0, ptr + 6, ptr + 30};
  EXPECT_EQ(static_ptrs, expected);
  EXPECT_EQ(dynamic_ptrs, expected);
}

TEST(TensorListShapeTest, Size) {
  TensorListShape<> tls_0(
    std::vector<TensorShape<DynamicDimensions>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});
  TensorListShape<3> tls_1({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});
  ASSERT_EQ(tls_0.size(), 3);
  ASSERT_EQ(tls_1.size(), 3);
  TensorListShape<> resized_empty{};
  resized_empty.resize(10);
  ASSERT_EQ(resized_empty.size(), 10);
  ASSERT_EQ(resized_empty.sample_dim(), 0);
  ASSERT_EQ(resized_empty.shapes, std::vector<int64_t>());

  TensorListShape<> empty_1{}, empty_2{std::vector<std::vector<int64_t>>{{}, {}}};
  ASSERT_EQ(empty_1.size(), 0);
  ASSERT_EQ(empty_2.size(), 2);

  TensorListShape<3> empty_3{}, one_zeros_static{std::vector<TensorShape<3>>{{}}};
  ASSERT_EQ(empty_3.size(), 0);
  ASSERT_TRUE(empty_3.shapes.empty());
  ASSERT_EQ(one_zeros_static.size(), 1);
  ASSERT_EQ(one_zeros_static.shapes, std::vector<int64_t>(3, 0));
  TensorListShape<0> empty_5{}, single_empty_6{std::vector<TensorShape<0>>{{}}};
  ASSERT_EQ(empty_5.size(), 0);
  ASSERT_EQ(single_empty_6.size(), 1);
}

TEST(TensorListShapeTest, Constructors) {
  TensorListShape<> tls_5samples(5), tls_5samples_2dim(5, 2);
  EXPECT_EQ(tls_5samples.size(), 5);
  EXPECT_EQ(tls_5samples.sample_dim(), 0);
  EXPECT_EQ(tls_5samples_2dim.size(), 5);
  EXPECT_EQ(tls_5samples_2dim.sample_dim(), 2);
  EXPECT_EQ(tls_5samples_2dim, uniform_list_shape(5, {0, 0}));
  // Check if we do not fall into calling initializer_list constructor for uniform initialization
  TensorListShape<> tls_5samples_brace{5}, tls_5samples_2dim_brace{5, 2};
  EXPECT_EQ(tls_5samples, tls_5samples_brace);
  EXPECT_EQ(tls_5samples_2dim, tls_5samples_2dim_brace);

  TensorListShape<2> tls_5samples_static2dim(5), tls_5samples_static2dim_2(5, 2);
  EXPECT_EQ(tls_5samples_static2dim.size(), 5);
  EXPECT_EQ(tls_5samples_static2dim.sample_dim(), 2);
  EXPECT_EQ(tls_5samples_static2dim, uniform_list_shape(5, {0, 0}));
  EXPECT_EQ(tls_5samples_static2dim_2.size(), 5);
  EXPECT_EQ(tls_5samples_static2dim_2.sample_dim(), 2);
  EXPECT_EQ(tls_5samples_static2dim_2, uniform_list_shape(5, {0, 0}));
}

TEST(TensorListShapeTest, ConstructorInitializerList) {
  auto vec = std::vector<int64_t>{1, 2, 3, 4};

  TensorListShape<> tls_0{{1, 2}, {3, 4}};
  EXPECT_EQ(tls_0.size(), 2);
  EXPECT_EQ(tls_0.sample_dim(), 2);
  EXPECT_EQ(tls_0.shapes, vec);

  TensorListShape<> tls_1({{1, 2}, {3, 4}});
  EXPECT_EQ(tls_1.size(), 2);
  EXPECT_EQ(tls_1.sample_dim(), 2);
  EXPECT_EQ(tls_1.shapes, vec);


  auto vec_2 = std::vector<int64_t>{1, 2};
  TensorListShape<> tls_2({{1}, {2}});
  EXPECT_EQ(tls_2.size(), 2);
  EXPECT_EQ(tls_2.sample_dim(), 1);
  EXPECT_EQ(tls_2.shapes, vec_2);

  auto vec_3 = std::vector<int64_t>{1, 2, 3, 4};
  TensorListShape<> tls_3 = {{TensorShape<>{1, 2}, TensorShape<>{3, 4}}};
  EXPECT_EQ(tls_3.size(), 2);
  EXPECT_EQ(tls_3.sample_dim(), 2);
  EXPECT_EQ(tls_3.shapes, vec_3);
}

TEST(TensorListShapeTest, Resize) {
  TensorListShape<> tls_0, tls_1(5, 2);
  tls_0.resize(5);
  EXPECT_EQ(tls_0.size(), 5);
  EXPECT_EQ(tls_0.sample_dim(), 0);
  EXPECT_TRUE(tls_0.shapes.empty());

  tls_0.resize(5, 2);
  EXPECT_EQ(tls_0.size(), 5);
  EXPECT_EQ(tls_0.sample_dim(), 2);
  EXPECT_EQ(tls_0.shapes, std::vector<int64_t>(10, 0));

  tls_1.resize(5, 3);
  EXPECT_EQ(tls_1.size(), 5);
  EXPECT_EQ(tls_1.sample_dim(), 3);
  EXPECT_EQ(tls_1.shapes, std::vector<int64_t>(15, 0));
  tls_1.resize(6, 3);
  EXPECT_EQ(tls_1.size(), 6);
  EXPECT_EQ(tls_1.sample_dim(), 3);
  EXPECT_EQ(tls_1.shapes, std::vector<int64_t>(18, 0));

  TensorListShape<2> tls_2, tls_3(5);
  tls_2.resize(5);
  EXPECT_EQ(tls_2.size(), 5);
  EXPECT_EQ(tls_2.sample_dim(), 2);
  EXPECT_EQ(tls_2.shapes, std::vector<int64_t>(10, 0));

  tls_1.resize(6, 2);
  EXPECT_EQ(tls_1.size(), 6);
  EXPECT_EQ(tls_1.sample_dim(), 2);
  EXPECT_EQ(tls_1.shapes, std::vector<int64_t>(12, 0));
}

TEST(TensorListShapeTest, ConstructorsFromVector) {
  auto shapes = std::vector<int64_t>{0, 1, 2, 3, 4, 5};

  TensorListShape<> tls_dyn_2_3(shapes, 2, 3);
  EXPECT_EQ(tls_dyn_2_3.shapes, shapes);
  EXPECT_EQ(tls_dyn_2_3.size(), 2);
  EXPECT_EQ(tls_dyn_2_3.sample_dim(), 3);

  TensorListShape<3> tls_static_2_3(shapes, 2, 3);
  EXPECT_EQ(tls_static_2_3.shapes, shapes);
  EXPECT_EQ(tls_static_2_3.size(), 2);
  EXPECT_EQ(tls_static_2_3.sample_dim(), 3);
}

TEST(TensorListShapeTest, IsUniform) {
  TensorListShape<3> non_uniform({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});
  EXPECT_FALSE(is_uniform(non_uniform));
  TensorListShape<3> uniform({{1, 2, 3}, {1, 2, 3}, {1, 2, 3}, {1, 2, 3}});
  EXPECT_TRUE(is_uniform(uniform));
  TensorListShape<3> empty;
  EXPECT_TRUE(is_uniform(empty));
}

TEST(TensorListShapeTest, Comparisons) {
  TensorListShape<3> empty_1, empty_2;
  EXPECT_TRUE(empty_1 == empty_2);
  TensorListShape<> empty_3, empty_4;
  EXPECT_TRUE(empty_3 == empty_4);

  TensorListShape<3> tls_1({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});
  EXPECT_TRUE(tls_1 == tls_1);
  TensorListShape<> tls_2(tls_1);
  EXPECT_TRUE(tls_1 == tls_2);

  TensorListShape<0> dim_0_10_0, dim_0_10_1, dim_0_11;
  dim_0_10_0.resize(10);
  dim_0_10_1.resize(10);
  dim_0_11.resize(11);
  EXPECT_TRUE(dim_0_10_0 == dim_0_10_1);
  EXPECT_FALSE(dim_0_10_0 == dim_0_11);
}

TEST(TensorListShapeTest, FirstStaticFromStatic) {
  TensorListShape<3> tls({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});

  TensorListShape<0> expected_static_0(std::vector<TensorShape<0>>{{}, {}, {}});
  TensorListShape<1> expected_static_1(std::vector<TensorShape<1>>{{1}, {2}, {3}});
  TensorListShape<2> expected_static_2(std::vector<TensorShape<2>>{{1, 2}, {2, 3}, {3, 4}});
  TensorListShape<3> expected_static_3(
      std::vector<TensorShape<3>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});

  auto first_0 = tls.first<0>();
  auto first_1 = tls.first<1>();
  auto first_2 = tls.first<2>();
  auto first_3 = tls.first<3>();

  EXPECT_EQ(first_0, expected_static_0);
  EXPECT_EQ(first_1, expected_static_1);
  EXPECT_EQ(first_2, expected_static_2);
  EXPECT_EQ(first_3, expected_static_3);

  EXPECT_EQ(first_0.size(), expected_static_0.size());
  EXPECT_EQ(first_1.size(), expected_static_1.size());
  EXPECT_EQ(first_2.size(), expected_static_2.size());
  EXPECT_EQ(first_3.size(), expected_static_3.size());
}

TEST(TensorListShapeTest, FirstStaticFromDynamic) {
  TensorListShape<> tls(
      std::vector<TensorShape<DynamicDimensions>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});

  TensorListShape<0> expected_static_0(std::vector<TensorShape<0>>{{}, {}, {}});
  TensorListShape<1> expected_static_1(std::vector<TensorShape<1>>{{1}, {2}, {3}});
  TensorListShape<2> expected_static_2(std::vector<TensorShape<2>>{{1, 2}, {2, 3}, {3, 4}});
  TensorListShape<3> expected_static_3(
      std::vector<TensorShape<3>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});

  auto first_0 = tls.first<0>();
  auto first_1 = tls.first<1>();
  auto first_2 = tls.first<2>();
  auto first_3 = tls.first<3>();

  EXPECT_EQ(first_0, expected_static_0);
  EXPECT_EQ(first_1, expected_static_1);
  EXPECT_EQ(first_2, expected_static_2);
  EXPECT_EQ(first_3, expected_static_3);

  EXPECT_EQ(first_0.size(), expected_static_0.size());
  EXPECT_EQ(first_1.size(), expected_static_1.size());
  EXPECT_EQ(first_2.size(), expected_static_2.size());
  EXPECT_EQ(first_3.size(), expected_static_3.size());
}

TEST(TensorListShapeTest, FirstDynamicFromStatic) {
  TensorListShape<3> tls({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});

  TensorListShape<0> expected_static_0(std::vector<TensorShape<0>>{{}, {}, {}});
  TensorListShape<1> expected_static_1(std::vector<TensorShape<1>>{{1}, {2}, {3}});
  TensorListShape<2> expected_static_2(std::vector<TensorShape<2>>{{1, 2}, {2, 3}, {3, 4}});
  TensorListShape<3> expected_static_3(
      std::vector<TensorShape<3>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});

  auto first_0 = tls.first(0);
  auto first_1 = tls.first(1);
  auto first_2 = tls.first(2);
  auto first_3 = tls.first(3);

  EXPECT_EQ(first_0, expected_static_0);
  EXPECT_EQ(first_1, expected_static_1);
  EXPECT_EQ(first_2, expected_static_2);
  EXPECT_EQ(first_3, expected_static_3);

  EXPECT_EQ(first_0.size(), expected_static_0.size());
  EXPECT_EQ(first_1.size(), expected_static_1.size());
  EXPECT_EQ(first_2.size(), expected_static_2.size());
  EXPECT_EQ(first_3.size(), expected_static_3.size());
}

TEST(TensorListShapeTest, FirstDynamicFromDynamic) {
  TensorListShape<> tls(
      std::vector<TensorShape<DynamicDimensions>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});

  TensorListShape<0> expected_static_0(std::vector<TensorShape<0>>{{}, {}, {}});
  TensorListShape<1> expected_static_1(std::vector<TensorShape<1>>{{1}, {2}, {3}});
  TensorListShape<2> expected_static_2(std::vector<TensorShape<2>>{{1, 2}, {2, 3}, {3, 4}});
  TensorListShape<3> expected_static_3(
      std::vector<TensorShape<3>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});

  auto first_0 = tls.first(0);
  auto first_1 = tls.first(1);
  auto first_2 = tls.first(2);
  auto first_3 = tls.first(3);

  EXPECT_EQ(first_0, expected_static_0);
  EXPECT_EQ(first_1, expected_static_1);
  EXPECT_EQ(first_2, expected_static_2);
  EXPECT_EQ(first_3, expected_static_3);

  EXPECT_EQ(first_0.size(), expected_static_0.size());
  EXPECT_EQ(first_1.size(), expected_static_1.size());
  EXPECT_EQ(first_2.size(), expected_static_2.size());
  EXPECT_EQ(first_3.size(), expected_static_3.size());
}

TEST(TensorListShapeTest, LastStaticFromStatic) {
  TensorListShape<3> tls({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});

  TensorListShape<0> expected_static_0(std::vector<TensorShape<0>>{{}, {}, {}});
  TensorListShape<1> expected_static_1(std::vector<TensorShape<1>>{{3}, {4}, {5}});
  TensorListShape<2> expected_static_2(std::vector<TensorShape<2>>{{2, 3}, {3, 4}, {4, 5}});
  TensorListShape<3> expected_static_3(
      std::vector<TensorShape<3>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});

  auto last_0 = tls.last<0>();
  auto last_1 = tls.last<1>();
  auto last_2 = tls.last<2>();
  auto last_3 = tls.last<3>();

  EXPECT_EQ(last_0, expected_static_0);
  EXPECT_EQ(last_1, expected_static_1);
  EXPECT_EQ(last_2, expected_static_2);
  EXPECT_EQ(last_3, expected_static_3);

  EXPECT_EQ(last_0.size(), expected_static_0.size());
  EXPECT_EQ(last_1.size(), expected_static_1.size());
  EXPECT_EQ(last_2.size(), expected_static_2.size());
  EXPECT_EQ(last_3.size(), expected_static_3.size());
}

TEST(TensorListShapeTest, LastStaticFromDynamic) {
  TensorListShape<> tls(
      std::vector<TensorShape<DynamicDimensions>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});

  TensorListShape<0> expected_static_0(std::vector<TensorShape<0>>{{}, {}, {}});
  TensorListShape<1> expected_static_1(std::vector<TensorShape<1>>{{3}, {4}, {5}});
  TensorListShape<2> expected_static_2(std::vector<TensorShape<2>>{{2, 3}, {3, 4}, {4, 5}});
  TensorListShape<3> expected_static_3(
      std::vector<TensorShape<3>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});

  auto last_0 = tls.last<0>();
  auto last_1 = tls.last<1>();
  auto last_2 = tls.last<2>();
  auto last_3 = tls.last<3>();

  EXPECT_EQ(last_0, expected_static_0);
  EXPECT_EQ(last_1, expected_static_1);
  EXPECT_EQ(last_2, expected_static_2);
  EXPECT_EQ(last_3, expected_static_3);

  EXPECT_EQ(last_0.size(), expected_static_0.size());
  EXPECT_EQ(last_1.size(), expected_static_1.size());
  EXPECT_EQ(last_2.size(), expected_static_2.size());
  EXPECT_EQ(last_3.size(), expected_static_3.size());
}

TEST(TensorListShapeTest, LastDynamicFromStatic) {
  TensorListShape<3> tls({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});

  TensorListShape<0> expected_static_0(std::vector<TensorShape<0>>{{}, {}, {}});
  TensorListShape<1> expected_static_1(std::vector<TensorShape<1>>{{3}, {4}, {5}});
  TensorListShape<2> expected_static_2(std::vector<TensorShape<2>>{{2, 3}, {3, 4}, {4, 5}});
  TensorListShape<3> expected_static_3(
      std::vector<TensorShape<3>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});

  auto last_0 = tls.last(0);
  auto last_1 = tls.last(1);
  auto last_2 = tls.last(2);
  auto last_3 = tls.last(3);

  EXPECT_EQ(last_0, expected_static_0);
  EXPECT_EQ(last_1, expected_static_1);
  EXPECT_EQ(last_2, expected_static_2);
  EXPECT_EQ(last_3, expected_static_3);

  EXPECT_EQ(last_0.size(), expected_static_0.size());
  EXPECT_EQ(last_1.size(), expected_static_1.size());
  EXPECT_EQ(last_2.size(), expected_static_2.size());
  EXPECT_EQ(last_3.size(), expected_static_3.size());
}

TEST(TensorListShapeTest, LastDynamicFromDynamic) {
  TensorListShape<> tls(
      std::vector<TensorShape<DynamicDimensions>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});

  TensorListShape<0> expected_static_0(std::vector<TensorShape<0>>{{}, {}, {}});
  TensorListShape<1> expected_static_1(std::vector<TensorShape<1>>{{3}, {4}, {5}});
  TensorListShape<2> expected_static_2(std::vector<TensorShape<2>>{{2, 3}, {3, 4}, {4, 5}});
  TensorListShape<3> expected_static_3(
      std::vector<TensorShape<3>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});

  auto last_0 = tls.last(0);
  auto last_1 = tls.last(1);
  auto last_2 = tls.last(2);
  auto last_3 = tls.last(3);

  EXPECT_EQ(last_0, expected_static_0);
  EXPECT_EQ(last_1, expected_static_1);
  EXPECT_EQ(last_2, expected_static_2);
  EXPECT_EQ(last_3, expected_static_3);

  EXPECT_EQ(last_0.size(), expected_static_0.size());
  EXPECT_EQ(last_1.size(), expected_static_1.size());
  EXPECT_EQ(last_2.size(), expected_static_2.size());
  EXPECT_EQ(last_3.size(), expected_static_3.size());
}

TEST(TensorListShapeTest, ToStatic) {
  TensorListShape<> tls_dynamic(
      std::vector<TensorShape<DynamicDimensions>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});
  auto tls_static = tls_dynamic.to_static<3>();
  EXPECT_EQ(tls_static, tls_dynamic);
  static_assert(std::is_same<decltype(tls_static), TensorListShape<3>>::value, "Wrong type");
  auto copy(tls_dynamic);
  auto moved_static = std::move(copy).to_static<3>();
  EXPECT_EQ(moved_static, tls_dynamic);
  static_assert(std::is_same<decltype(moved_static), TensorListShape<3>>::value, "Wrong type");
}

template <int out_dim, int in_dim>
void TestConvertDim(const TensorShape<in_dim> &in) {
  auto o = convert_dim<out_dim>(in);
  EXPECT_EQ(o.size(), out_dim);
  EXPECT_EQ(o, in);
  static_assert(std::is_same<TensorShape<out_dim>, decltype(o)>::value,
                "convert_dim<out_dim> produced unexpected type");

  auto odyn = convert_dim<DynamicDimensions>(in);
  static_assert(std::is_same<TensorShape<>, decltype(odyn)>::value,
                "convert_dim<DynamicDimension> produced unexpected type");
  EXPECT_EQ(odyn.size(), out_dim);
  EXPECT_EQ(odyn, in);
}

TEST(TensorShapeTest, ConvertDim) {
  TensorShape<> i0 = { 1, 2, 3, 4, 5 };
  TensorShape<1> i1 = { 1 };
  TensorShape<2> i2 = { 1, 2 };
  TensorShape<3> i3 = { 1, 2, 3 };
  TensorShape<4> i4 = { 1, 2, 3, 4 };
  TestConvertDim<5>(i0);
  TestConvertDim<1>(i1);
  TestConvertDim<2>(i2);
  TestConvertDim<3>(i3);
  TestConvertDim<4>(i4);
}

template <int out_dim, int in_dim = out_dim>
void TestConvertDim() {
  unsigned seed = 123;
  std::vector<TensorShape<in_dim>> shapes;
  for (int i = 0; i < 10; i++) {
    TensorShape<in_dim> shape;
    shape.resize(out_dim);
    for (int j = 0; j < out_dim; j++) {
      shape[j] = rand_r(&seed)%10;
    }
    shapes.push_back(shape);
  }
  TensorListShape<in_dim> in(shapes);

  auto o = convert_dim<out_dim>(in);
  EXPECT_EQ(o.sample_dim(), out_dim);
  EXPECT_EQ(o, in);
  static_assert(std::is_same<TensorListShape<out_dim>, decltype(o)>::value,
                "convert_dim<out_dim> produced unexpected type");
  auto ptr = in.shapes.data();
  auto o2 = convert_dim<out_dim>(std::move(in));
  EXPECT_TRUE(in.shapes.empty());
  EXPECT_EQ(o2.shapes.data(), ptr);

  in = {shapes};

  auto odyn = convert_dim<DynamicDimensions>(in);
  static_assert(std::is_same<TensorListShape<>, decltype(odyn)>::value,
                "convert_dim<DynamicDimension> produced unexpected type");
  EXPECT_EQ(odyn.sample_dim(), out_dim);
  EXPECT_EQ(odyn, in);

  ptr = in.shapes.data();
  auto odyn2 = convert_dim<DynamicDimensions>(std::move(in));
  EXPECT_TRUE(in.shapes.empty());
  EXPECT_EQ(odyn2.shapes.data(), ptr);
}

TEST(TensorListShapeTest, ConvertDim) {
  TestConvertDim<5, -1>();
  TestConvertDim<1>();
  TestConvertDim<2>();
  TestConvertDim<3>();
  TestConvertDim<4>();
}

using shape_blocks_t = SmallVector<std::pair<int, int>, 6>;

TEST(TensorShapeTest, CollapseDimsEmpty) {
  auto groups = shape_blocks_t{};
  auto result = collapse_dims(TensorShape<>{}, make_cspan(groups));
  auto expected = TensorShape<>{};
  EXPECT_EQ(expected, result);
}

TEST(TensorShapeTest, CollapseDimsOneElem) {
  auto groups = shape_blocks_t{{0, 1}};
  auto result = collapse_dims(TensorShape<>{1}, make_cspan(groups));
  auto expected = TensorShape<>{1};
  EXPECT_EQ(expected, result);
}

TEST(TensorShapeTest, CollapseDimsTwoElems) {
  auto groups_0 = shape_blocks_t{{0, 2}};
  auto result_0 = collapse_dims(TensorShape<>{1, 2}, make_cspan(groups_0));
  auto expected_0 = TensorShape<>{2};
  EXPECT_EQ(expected_0, result_0);

  auto groups_1 = shape_blocks_t{{0, 1}, {1, 1}};
  auto result_1 = collapse_dims(TensorShape<>{1, 2}, make_cspan(groups_1));
  auto expected_1 = TensorShape<>{1, 2};
  EXPECT_EQ(expected_1, result_1);
}

TEST(TensorShapeTest, CollapseDims) {
  auto groups_0 = shape_blocks_t{{0, 3}};
  auto result_0 = collapse_dims(TensorShape<>{1, 2, 3}, make_cspan(groups_0));
  auto expected_0 = TensorShape<>{6};
  EXPECT_EQ(expected_0, result_0);

  auto groups_1 = shape_blocks_t{{0, 2}, {2, 1}};
  auto result_1 = collapse_dims(TensorShape<>{1, 2, 3}, make_cspan(groups_1));
  auto expected_1 = TensorShape<>{2, 3};
  EXPECT_EQ(expected_1, result_1);

  auto groups_2 = shape_blocks_t{{0, 2}, {2, 1}, {3, 3}};
  auto result_2 = collapse_dims(TensorShape<>{1, 2, 3, 4, 5, 6}, make_cspan(groups_2));
  auto expected_2 = TensorShape<>{2, 3, 120};
  EXPECT_EQ(expected_2, result_2);

  auto groups_3 = shape_blocks_t{{0, 3}, {3, 1}};
  auto result_3 = collapse_dims(TensorShape<>{1, 2, 3, 4}, make_cspan(groups_3));
  auto expected_3 = TensorShape<>{6, 4};
  EXPECT_EQ(expected_3, result_3);
}

TEST(TensorShapeTest, CollapseDimsSkip) {
  auto groups_0 = shape_blocks_t{};
  auto result_0 = collapse_dims(TensorShape<>{1, 2, 3}, make_cspan(groups_0));
  auto expected_0 = TensorShape<>{1, 2, 3};
  EXPECT_EQ(expected_0, result_0);

  auto groups_1 = shape_blocks_t{{0, 1}};
  auto result_1 = collapse_dims(TensorShape<>{1, 2, 3}, make_cspan(groups_1));
  auto expected_1 = TensorShape<>{1, 2, 3};
  EXPECT_EQ(expected_1, result_1);

  auto groups_2 = shape_blocks_t{{1, 1}};
  auto result_2 = collapse_dims(TensorShape<>{1, 2, 3}, make_cspan(groups_2));
  auto expected_2 = TensorShape<>{1, 2, 3};
  EXPECT_EQ(expected_2, result_2);

  auto groups_3 = shape_blocks_t{{2, 1}};
  auto result_3 = collapse_dims(TensorShape<>{1, 2, 3}, make_cspan(groups_3));
  auto expected_3 = TensorShape<>{1, 2, 3};
  EXPECT_EQ(expected_3, result_3);

  auto groups_4 = shape_blocks_t{{0, 1}, {1, 1}, {2, 1}};
  auto result_4 = collapse_dims(TensorShape<>{1, 2, 3}, make_cspan(groups_4));
  auto expected_4 = TensorShape<>{1, 2, 3};
  EXPECT_EQ(expected_4, result_4);

  auto groups_5 = shape_blocks_t{{0, 1}, {2, 1}};
  auto result_5 = collapse_dims(TensorShape<>{1, 2, 3}, make_cspan(groups_5));
  auto expected_5 = TensorShape<>{1, 2, 3};
  EXPECT_EQ(expected_5, result_5);

  auto groups_6 = shape_blocks_t{{0, 2}, {3, 3}};
  auto result_6 = collapse_dims(TensorShape<>{1, 2, 3, 4, 5, 6}, make_cspan(groups_6));
  auto expected_6 = TensorShape<>{2, 3, 120};
  EXPECT_EQ(expected_6, result_6);

  auto groups_7 = shape_blocks_t{{0, 2}, {3, 3}};
  auto result_7 = collapse_dims(TensorShape<>{1, 2, 3, 4, 5, 6, 7}, make_cspan(groups_7));
  auto expected_7 = TensorShape<>{2, 3, 120, 7};
  EXPECT_EQ(expected_7, result_7);
}

TEST(TensorShapeTest, CollapseDimsStatic) {
  auto perm_0 = std::vector<int>{0, 1, 2};
  auto groups_0 = shape_blocks_t{{0, 3}};
  auto result_0 = collapse_dims<1>(TensorShape<3>{1, 2, 3}, make_cspan(groups_0));
  auto expected_0 = TensorShape<>{6};
  EXPECT_EQ(expected_0, result_0);

  auto groups_1 = shape_blocks_t{{0, 2}, {2, 1}};
  auto result_1 = collapse_dims<2>(TensorShape<3>{1, 2, 3}, make_cspan(groups_1));
  auto expected_1 = TensorShape<>{2, 3};
  EXPECT_EQ(expected_1, result_1);

  auto groups_2 = shape_blocks_t{{0, 2}, {2, 1}, {3, 3}};
  auto result_2 = collapse_dims<3>(TensorShape<6>{1, 2, 3, 4, 5, 6}, make_cspan(groups_2));
  auto expected_2 = TensorShape<>{2, 3, 120};
  EXPECT_EQ(expected_2, result_2);

  auto groups_3 = shape_blocks_t{{0, 3}, {3, 1}};
  auto result_3 = collapse_dims<2>(TensorShape<4>{1, 2, 3, 4}, make_cspan(groups_3));
  auto expected_3 = TensorShape<>{6, 4};
  EXPECT_EQ(expected_3, result_3);
}

TEST(TensorListShapeTest, IsDegenerateDim) {
  TensorListShape<> tls = {
    { 1, 1, 3, 1, 1 },
    { 1, 2, 1, 1, 5 }
  //  T  F  F  T  F    <-- all samples with extent == 1
  };
  ASSERT_EQ(tls.sample_dim(), 5);
  EXPECT_TRUE(is_degenerate_dim(tls, 0));
  EXPECT_FALSE(is_degenerate_dim(tls, 1));
  EXPECT_FALSE(is_degenerate_dim(tls, 2));
  EXPECT_TRUE(is_degenerate_dim(tls, 3));
  EXPECT_FALSE(is_degenerate_dim(tls, 4));
}

TEST(TensorListShapeTest, CollapseDim) {
  TensorListShape<7> in = {{
    { 2, 4, 3, 2, 1, 3, 5 },
    { 2, 1, 3, 6, 7, 3, 4 }
  }};
  TensorListShape<> in_dyn = in;
  TensorListShape<6> ref0 = {{
    { 8, 3, 2, 1, 3, 5 },
    { 2, 3, 6, 7, 3, 4 }
  }};
  TensorListShape<> ref1 = {{
    { 2, 12, 2, 1, 3, 5 },
    { 2, 3, 6, 7, 3, 4 }
  }};
  TensorListShape<> ref5 = {{
    { 2, 4, 3, 2, 1, 15 },
    { 2, 1, 3, 6, 7, 12 }
  }};
  TensorListShape<6> out0;
  collapse_dim(out0, in, 0);
  EXPECT_EQ(out0, ref0);
  TensorListShape<> out1 = collapse_dim(in_dyn, 1);
  EXPECT_EQ(out1, ref1);
  TensorListShape<6> out5 = collapse_dim(in, 5);
  EXPECT_EQ(out5, ref5);
}

TEST(TensorListShapeTest, CollapseDims_Noop) {
  TensorListShape<> in = {{
    { 2, 4, 3, 2, 1, 3, 5 },
    { 2, 1, 3, 6, 7, 3, 4 }
  //  +--+     +-----+
  }};
  TensorListShape<> out = collapse_dims(in, {});
  EXPECT_EQ(out, in);
}

TEST(TensorListShapeTest, CollapseDims_Dyn) {
  TensorListShape<> in = {{
    { 2, 4, 3, 2, 1, 3, 5 },
    { 2, 1, 3, 6, 7, 3, 4 }
  //  +--+     +-----+
  }};
  TensorListShape<> ref = {{
    {  8,   3,   6,     5 },
    {  2,   3,  126,    4 }
  }};
  TensorListShape<> out = collapse_dims(in, { { 0, 2 }, { 3, 3 } });
  EXPECT_EQ(out, ref);
  TensorListShape<4> out_static = collapse_dims<4>(in, { { 0, 2 }, { 2, 1 }, { 3, 3 }, { 6, 1 } });
  EXPECT_EQ(out_static, ref);
}

TEST(TensorListShapeTest, CollapseDims_Static) {
  TensorListShape<7> in = {{
    { 2, 4, 3, 2, 1, 3, 5 },
    { 2, 1, 3, 6, 7, 3, 4 }
  //  +--+     +-----+
  }};
  TensorListShape<4> ref = {{
    {  8,   3,   6,     5 },
    {  2,   3,  126,    4 }
  }};
  TensorListShape<4> out = collapse_dims<4>(in, { { 0, 2 }, { 3, 3 } });
  EXPECT_EQ(out, ref);
  auto out_dyn = collapse_dims(in, { { 0, 2 }, { 2, 1 }, { 3, 3 }, { 6, 1 } });
  EXPECT_EQ(out_dyn, ref);
}

TEST(TensorListShapeTest, PermuteSamples) {
  TensorListShape<> tl = {{
    { 1, 2 }, { 3, 4 }, { 5, 6 }
  }};
  TensorListShape<2> ref = {{
    { 5, 6 }, { 1, 2 }, { 3, 4 }
  }};
  int perm[] = { 2, 0, 1 };
  auto permuted = permute_samples<2>(tl, perm);
  EXPECT_EQ(ref, permuted) << "Actual:\n" << permuted << "\nexpected:\n" << ref;
}

TEST(TensorListShapeTest, PermuteDims) {
  TensorListShape<4> tl = {{
    { 1, 2, 3, 4 }, { 5, 6, 7, 8 }
  }};
  TensorListShape<4> ref = {{
    { 3, 2, 4, 1 }, { 7, 6, 8, 5 }
  }};
  int perm[] = { 2, 1, 3, 0 };
  auto permuted = permute_dims(tl, perm);
  EXPECT_EQ(ref, permuted) << "Actual:\n" << permuted << "\nexpected:\n" << ref;
}

TEST(TensorTest, WontCompile) {
  // TensorShape<5> static_shape_less(1, 2, 3, 4);
  // TensorShape<5> static_shape_more(1, 2, 3, 4, 5, 6);
  // TensorShape<DynamicDimensions>().to_static<DynamicDimensions>();
  // TensorShape<5>{TensorShape<DynamicDimensions>()};
  // TensorShape<-2> negative;

  // TensorView<EmptyBackendTag, int8_t, 4>(static_cast<int*>(nullptr), {1, 2, 3, 4});
  // TensorView<EmptyBackendTag, int, 4>{TensorView<EmptyBackendTag, int, DynamicDimensions>{}};
  // TensorView<EmptyBackendTag, int, DynamicDimensions>{TensorView<EmptyBackendTag, int8_t, 4>{}};
}


TEST(AppendTest, NonuniformShapes) {
  TensorListShape<4> tls0 = {{{1, 2, 3, 4}, {5, 6, 7, 8}}};
  TensorListShape<-1> tls1 = {{{1, 2, 3, 4}, {5, 6, 7, 8}}};
  TensorListShape<-1> tls2 = {{{9, 10, 11}, {13, 14, 15}}};
  TensorListShape<3> tls3 = {{{17, 18, 19}, {22, 23, 24}}};
  TensorListShape<4> tls4 = {{{17, 18, 19, 20}, {21, 22, 23, 24}}};

  EXPECT_THROW(tls1.append({tls3, tls4}), std::invalid_argument);
  EXPECT_THROW(tls1.append({tls1, tls3}), std::invalid_argument);
  EXPECT_THROW(tls1.append({tls1, tls2}), std::invalid_argument);
  EXPECT_THROW(tls1.append(tls2), std::invalid_argument);
}


TEST(AppendTest, AppendToEmpty) {
  TensorListShape<4> tls1 = {{{1, 2, 3, 4}, {5, 6, 7, 8}}};
  TensorListShape<-1> tls2 = {{{9, 10, 11, 12}, {13, 14, 15, 16}}};
  TensorListShape<4> tls3 = {{{17, 18, 19, 20}, {21, 22, 23, 24}}};
  TensorListShape<4> ref = {{
    {1, 2, 3, 4}, {5, 6, 7, 8},
    {9, 10, 11, 12}, {13, 14, 15, 16},
    {17, 18, 19, 20}, {21, 22, 23, 24}
  }};
  TensorListShape<-1> empty_dyn;
  TensorListShape<4> empty_st;
  empty_st.append({tls1, tls2.to_static<4>(), tls3});
  empty_dyn.append({tls1, tls2, tls3});
  EXPECT_EQ(empty_st, ref);
  EXPECT_EQ(empty_dyn, ref);
}


TEST(AppendTest, AllEmpty) {
  TensorListShape<-1> tls_tested_dyn;
  TensorListShape<-1> empty_dyn;
  TensorListShape<4> tls_tested_st;
  TensorListShape<4> empty_st;
  TensorListShape<4> ref_st;
  TensorListShape<-1> ref_dyn;

  tls_tested_st.append({empty_st, empty_dyn.to_static<4>()});
  tls_tested_dyn.append({empty_st, empty_dyn});
  EXPECT_EQ(tls_tested_st, ref_st);
  EXPECT_EQ(tls_tested_dyn, ref_dyn);
}


TEST(AppendTest, AppendEmpty) {
  TensorListShape<4> tls_tested_st = {{{1, 2, 3, 4}, {5, 6, 7, 8}}};
  TensorListShape<-1> tls_tested_dyn = {{{1, 2, 3, 4}, {5, 6, 7, 8}}};
  TensorListShape<4> ref = {{{1, 2, 3, 4}, {5, 6, 7, 8}}};
  TensorListShape<-1> empty_dyn;
  TensorListShape<4> empty_st;

  tls_tested_st.append({empty_st, empty_dyn.to_static<4>()});
  tls_tested_dyn.append({empty_st, empty_dyn});
  EXPECT_EQ(tls_tested_st, ref);
  EXPECT_EQ(tls_tested_dyn, ref);
}


TEST(AppendTest, AppendTest) {
  TensorListShape<4> tls_tested_st = {{{1, 2, 3, 4}, {5, 6, 7, 8}}};
  TensorListShape<-1> tls_tested_dyn = {{{1, 2, 3, 4}, {5, 6, 7, 8}}};
  TensorListShape<4> tls2 = {{{9, 10, 11, 12}, {13, 14, 15, 16}}};
  TensorListShape<-1> tls3 = {{{17, 18, 19, 20}, {21, 22, 23, 24}}};
  TensorListShape<4> ref = {{
    {1, 2, 3, 4}, {5, 6, 7, 8},
    {9, 10, 11, 12}, {13, 14, 15, 16},
    {17, 18, 19, 20}, {21, 22, 23, 24}
  }};

  tls_tested_st.append({tls2, tls3.to_static<4>()});
  tls_tested_dyn.append({tls2, tls3});
  EXPECT_EQ(tls_tested_st, ref);
  EXPECT_EQ(tls_tested_dyn, ref);
}

TEST(AppendTest, ZeroDim) {
  TensorListShape<0> tls_tested_st = {{}, 4, 0};
  TensorListShape<-1> tls_tested_dyn = {{}, 4, 0};
  TensorListShape<0> tls1 = {{}, 4, 0};
  TensorListShape<-1> tls2 = {{}, 4, 0};
  TensorListShape<0> tls3 = {{}, 4, 0};
  TensorListShape<0> ref = {{}, 12, 0};

  tls_tested_st.append({tls2.to_static<0>(), tls3});
  tls_tested_dyn.append({tls2, tls3});
  EXPECT_EQ(tls_tested_st, ref);
  EXPECT_EQ(tls_tested_dyn, ref);
}


TEST(AppendTest, AppendSingleTLS) {
  TensorListShape<4> tls_tested_st = {{{1, 2, 3, 4}, {5, 6, 7, 8}}};
  TensorListShape<-1> tls_tested_dyn = {{{1, 2, 3, 4}, {5, 6, 7, 8}}};
  TensorListShape<4> tls2 = {{{9, 10, 11, 12}, {13, 14, 15, 16}}};
  TensorListShape<-1> tls3 = {{{9, 10, 11, 12}, {13, 14, 15, 16}}};
  TensorListShape<4> ref = {{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}};

  tls_tested_st.append(tls2);
  tls_tested_dyn.append(tls3);
  EXPECT_EQ(tls_tested_st, ref);
  EXPECT_EQ(tls_tested_dyn, ref);
}


TEST(TensorListShapeTest, SampleRangeTest) {
  TensorListShape<> tls = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}, {8, 9}};
  TensorListShape<> ref091 = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5},
                              {5, 6}, {6, 7}, {7, 8}, {8, 9}};
  TensorListShape<> ref081 = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}};
  TensorListShape<> ref052 = {{0, 1}, {2, 3}, {4, 5}};
  TensorListShape<> ref183 = {{1, 2}, {4, 5}, {7, 8}};
  TensorListShape<> ref092 = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}};
  TensorListShape<> ref192 = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
  TensorListShape<> ref1917 = {{1, 2}};

  EXPECT_EQ(sample_range(tls, 0, 9, 1), ref091);
  EXPECT_EQ(sample_range(tls, 0, 8, 1), ref081);
  EXPECT_EQ(sample_range(tls, 0, 5, 2), ref052);
  EXPECT_EQ(sample_range(tls, 1, 8, 3), ref183);
  EXPECT_EQ(sample_range(tls, 0, 9, 2), ref092);
  EXPECT_EQ(sample_range(tls, 1, 9, 2), ref192);
  EXPECT_EQ(sample_range(tls, 1, 9, 17), ref1917);
  EXPECT_EQ(sample_range(tls, 9, 9, 17), TensorListShape<>(0, 2));
  EXPECT_EQ(sample_range(TensorListShape<>(), 0, 0), TensorListShape<>());
}

}  // namespace kernels
}  // namespace dali
