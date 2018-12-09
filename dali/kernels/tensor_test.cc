// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/kernels/shape.h"
#include "dali/kernels/tensor_view.h"

namespace tensor {

TEST(TensorShapeTest, StaticShapeConstructor) {
  TensorShape<0> zero_tensor;

  constexpr int test_dim = 5;
  std::array<int64_t, test_dim> test_shape = {1, 2, 3, 4, 5};

  // Default constructor
  TensorShape<test_dim> empty_tensor;
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(empty_tensor[i], int64_t(0));
  }

  // std::array and expanded list constructor
  TensorShape<test_dim> a(test_shape);
  TensorShape<test_dim> b(test_shape[0], test_shape[1], test_shape[2], test_shape[3], test_shape[4]);
  ASSERT_EQ(a.size(), test_dim);
  ASSERT_EQ(b.size(), test_dim);
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(a[i], test_shape[i]);
    ASSERT_EQ(b[i], test_shape[i]);
  }

  // Copy constructor
  TensorShape<test_dim> check_construct(a);
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(check_construct[i], a[i]);
  }

  // Assignement
  TensorShape<test_dim> check_assign;
  check_assign = a;
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(check_assign[i], a[i]);
  }

  // Move rvalue
  TensorShape<test_dim> check_move_construct(TensorShape<5>{test_shape});
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(check_move_construct[i], test_shape[i]);
  }

  // Assignement for rvalue
  TensorShape<test_dim> check_move_assign;
  check_move_assign = TensorShape<5>{test_shape};
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(check_move_assign[i], test_shape[i]);
  }

}

TEST(TensorShapeTest, DynamicShapeConstructor) {
  // Default
  TensorShape<DynamicDimensions> zero_tensor;
  ASSERT_EQ(zero_tensor.size(), 0);

  // std::array constructor
  constexpr int test_dim = 5;
  std::array<int64_t, test_dim> test_shape_arr = {1, 2, 3, 4, 5};
  TensorShape<DynamicDimensions> a(test_shape_arr);
  ASSERT_EQ(a.size(), test_dim);
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(a[i], test_shape_arr[i]);
  }


  // std::vector constructor
  std::vector<int64_t> test_shape_vec = {1, 2, 3, 4, 5, 6, 7};
  TensorShape<DynamicDimensions> b(test_shape_vec);
  ASSERT_EQ(b.size(), test_shape_vec.size());
  for (int i = 0; i < test_shape_vec.size(); i++) {
    ASSERT_EQ(b[i], test_shape_vec[i]);
  }

  // Expanded arguments constructor
  TensorShape<DynamicDimensions> c(1);
  ASSERT_EQ(c.size(), 1);
  ASSERT_EQ(c[0], 1);

  TensorShape<DynamicDimensions> d(1, 2, 3, 4);
  ASSERT_EQ(d.size(), 4);
  ASSERT_EQ(d[0], 1);
  ASSERT_EQ(d[1], 2);
  ASSERT_EQ(d[2], 3);
  ASSERT_EQ(d[3], 4);

  // Copy constructor
  TensorShape<DynamicDimensions> check_construct(a);
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(check_construct[i], a[i]);
  }

  // Asignement
  TensorShape<DynamicDimensions> check_assign;
  check_assign = a;
  ASSERT_EQ(check_assign.size(), a.size());
  for (int i = 0; i < a.size(); i++) {
    ASSERT_EQ(check_assign[i], a[i]);
  }

  // Second asignement to the same value
  check_assign = b;
  ASSERT_EQ(check_assign.size(), b.size());
  for (int i = 0; i < b.size(); i++) {
    ASSERT_EQ(check_assign[i], b[i]);
  }

    // Move rvalue
  TensorShape<DynamicDimensions> check_move_construct(TensorShape<DynamicDimensions>{test_shape_arr});
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(check_move_construct[i], test_shape_arr[i]);
  }

  // Assignement for rvalue
  TensorShape<DynamicDimensions> check_move_assign;
  check_move_assign = TensorShape<DynamicDimensions>{test_shape_arr};
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(check_move_assign[i], test_shape_arr[i]);
  }
}


TEST(TensorShapeTest, StaticDynamicConversions) {
  TensorShape<3> static_shape_3(1, 2, 3);
  TensorShape<5> static_shape_5(1, 2, 3, 4, 5);

  TensorShape<DynamicDimensions> check_construct_3(static_shape_3);
  ASSERT_EQ(check_construct_3.size(), static_shape_3.size());
  for (int i = 0; i < static_shape_3.size(); i++) {
    ASSERT_EQ(check_construct_3[i], static_shape_3[i]);
  }

  TensorShape<DynamicDimensions> check_construct_5(static_shape_5);
  ASSERT_EQ(check_construct_5.size(), static_shape_5.size());
  for (int i = 0; i < static_shape_5.size(); i++) {
    ASSERT_EQ(check_construct_5[i], static_shape_5[i]);
  }

  TensorShape<DynamicDimensions> check_assign;
  check_assign = static_shape_3;
  ASSERT_EQ(check_assign.size(), static_shape_3.size());
  for (int i = 0; i < static_shape_3.size(); i++) {
    ASSERT_EQ(check_assign[i], static_shape_3[i]);
  }

  check_assign = static_shape_5;
  ASSERT_EQ(check_assign.size(), static_shape_5.size());
  for (int i = 0; i < static_shape_5.size(); i++) {
    ASSERT_EQ(check_assign[i], static_shape_5[i]);
  }

  auto s3 = TensorShape<3>{2, 4, 6};
  check_assign = s3;
  static_shape_3 = check_assign.to_static_ndim<3>();
  for (int i = 0; i < s3.size(); i++) {
    ASSERT_EQ(static_shape_3[i], s3[i]);
  }
}

TEST(TensorShapeTest, Comparisons) {
  // Static ndim
  ASSERT_EQ(TensorShape<1>(1) == TensorShape<1>(1), true);
  ASSERT_EQ(TensorShape<1>(1) != TensorShape<1>(1), false);

  ASSERT_EQ(TensorShape<1>(1) == TensorShape<1>(2), false);
  ASSERT_EQ(TensorShape<1>(1) != TensorShape<1>(2), true);

  ASSERT_EQ(TensorShape<3>(1, 2, 3) == TensorShape<3>(1, 2, 3), true);
  ASSERT_EQ(TensorShape<3>(1, 2, 3) != TensorShape<3>(1, 2, 3), false);

  ASSERT_EQ(TensorShape<3>(1, 2, 3) == TensorShape<3>(1, 4, 3), false);
  ASSERT_EQ(TensorShape<3>(1, 2, 3) != TensorShape<3>(1, 4, 3), true);


  ASSERT_EQ(TensorShape<1>(1) == TensorShape<2>(1, 2), false);
  ASSERT_EQ(TensorShape<1>(1) != TensorShape<2>(1, 2), true);
  ASSERT_EQ(TensorShape<2>(1, 2) == TensorShape<1>(1), false);
  ASSERT_EQ(TensorShape<2>(1, 2) != TensorShape<1>(1), true);

  // Dynamic ndim
  ASSERT_EQ(TensorShape<DynamicDimensions>(1) == TensorShape<DynamicDimensions>(1), true);
  ASSERT_EQ(TensorShape<DynamicDimensions>(1) != TensorShape<DynamicDimensions>(1), false);

  ASSERT_EQ(TensorShape<DynamicDimensions>(1) == TensorShape<DynamicDimensions>(2), false);
  ASSERT_EQ(TensorShape<DynamicDimensions>(1) != TensorShape<DynamicDimensions>(2), true);

  ASSERT_EQ(TensorShape<DynamicDimensions>(1, 2, 3) == TensorShape<DynamicDimensions>(1, 2, 3), true);
  ASSERT_EQ(TensorShape<DynamicDimensions>(1, 2, 3) != TensorShape<DynamicDimensions>(1, 2, 3), false);

  ASSERT_EQ(TensorShape<DynamicDimensions>(1, 2, 3) == TensorShape<DynamicDimensions>(1, 4, 3), false);
  ASSERT_EQ(TensorShape<DynamicDimensions>(1, 2, 3) != TensorShape<DynamicDimensions>(1, 4, 3), true);

  ASSERT_EQ(TensorShape<DynamicDimensions>(1) == TensorShape<DynamicDimensions>(1, 2), false);
  ASSERT_EQ(TensorShape<DynamicDimensions>(1) != TensorShape<DynamicDimensions>(1, 2), true);
  ASSERT_EQ(TensorShape<DynamicDimensions>(1, 2) == TensorShape<DynamicDimensions>(1), false);
  ASSERT_EQ(TensorShape<DynamicDimensions>(1, 2) != TensorShape<DynamicDimensions>(1), true);

  // Mixed ndim
  ASSERT_EQ(TensorShape<1>(1) == TensorShape<DynamicDimensions>(1), true);
  ASSERT_EQ(TensorShape<1>(1) != TensorShape<DynamicDimensions>(1), false);
  ASSERT_EQ(TensorShape<DynamicDimensions>(1) == TensorShape<1>(1), true);
  ASSERT_EQ(TensorShape<DynamicDimensions>(1) != TensorShape<1>(1), false);

  ASSERT_EQ(TensorShape<1>(1) == TensorShape<DynamicDimensions>(2), false);
  ASSERT_EQ(TensorShape<1>(1) != TensorShape<DynamicDimensions>(2), true);
  ASSERT_EQ(TensorShape<DynamicDimensions>(1) == TensorShape<1>(2), false);
  ASSERT_EQ(TensorShape<DynamicDimensions>(1) != TensorShape<1>(2), true);

  ASSERT_EQ(TensorShape<3>(1, 2, 3) == TensorShape<DynamicDimensions>(1, 2, 3), true);
  ASSERT_EQ(TensorShape<3>(1, 2, 3) != TensorShape<DynamicDimensions>(1, 2, 3), false);
  ASSERT_EQ(TensorShape<DynamicDimensions>(1, 2, 3) == TensorShape<3>(1, 2, 3), true);
  ASSERT_EQ(TensorShape<DynamicDimensions>(1, 2, 3) != TensorShape<3>(1, 2, 3), false);

  ASSERT_EQ(TensorShape<3>(1, 2, 3) == TensorShape<DynamicDimensions>(1, 4, 3), false);
  ASSERT_EQ(TensorShape<3>(1, 2, 3) != TensorShape<DynamicDimensions>(1, 4, 3), true);
  ASSERT_EQ(TensorShape<DynamicDimensions>(1, 2, 3) == TensorShape<3>(1, 4, 3), false);
  ASSERT_EQ(TensorShape<DynamicDimensions>(1, 2, 3) != TensorShape<3>(1, 4, 3), true);

  ASSERT_EQ(TensorShape<1>(1) == TensorShape<DynamicDimensions>(1, 2), false);
  ASSERT_EQ(TensorShape<1>(1) != TensorShape<DynamicDimensions>(1, 2), true);
  ASSERT_EQ(TensorShape<2>(1, 2) == TensorShape<DynamicDimensions>(1), false);
  ASSERT_EQ(TensorShape<2>(1, 2) != TensorShape<DynamicDimensions>(1), true);
  ASSERT_EQ(TensorShape<DynamicDimensions>(1) == TensorShape<2>(1, 2), false);
  ASSERT_EQ(TensorShape<DynamicDimensions>(1) != TensorShape<2>(1, 2), true);
  ASSERT_EQ(TensorShape<DynamicDimensions>(1, 2) == TensorShape<1>(1), false);
  ASSERT_EQ(TensorShape<DynamicDimensions>(1, 2) != TensorShape<1>(1), true);

}

TEST(TensorShapeTest, RangeLoop) {
  TensorShape<5> ts{0, 1, 2, 3, 4};
  int expected = 0;
  for (auto s : ts) {
    ASSERT_EQ(s, expected);
    expected++;
  }
}

TEST(TensorShapeTest, FirstStatic) {
  TensorShape<5> ts(1, 2, 3, 4, 5);
  ASSERT_EQ(ts.first<0>(), TensorShape<0>());
  ASSERT_EQ(ts.first<1>(), TensorShape<1>(1));
  ASSERT_EQ(ts.first<2>(), TensorShape<2>(1, 2));
  ASSERT_EQ(ts.first<3>(), TensorShape<3>(1, 2, 3));
  ASSERT_EQ(ts.first<4>(), TensorShape<4>(1, 2, 3, 4));
  ASSERT_EQ(ts.first<5>(), TensorShape<5>(1, 2, 3, 4, 5));
}

TEST(TensorShapeTest, LastStatic) {
  TensorShape<5> ts(1, 2, 3, 4, 5);
  ASSERT_EQ(ts.last<0>(), TensorShape<0>());
  ASSERT_EQ(ts.last<1>(), TensorShape<1>(5));
  ASSERT_EQ(ts.last<2>(), TensorShape<2>(4, 5));
  ASSERT_EQ(ts.last<3>(), TensorShape<3>(3, 4, 5));
  ASSERT_EQ(ts.last<4>(), TensorShape<4>(2, 3, 4, 5));
  ASSERT_EQ(ts.last<5>(), TensorShape<5>(1, 2, 3, 4, 5));
}

TEST(TensorShapeTest, FirstDynamic) {
  TensorShape<DynamicDimensions> ts(1, 2, 3, 4, 5);
  ASSERT_EQ(ts.first(0), TensorShape<DynamicDimensions>());
  ASSERT_EQ(ts.first(1), TensorShape<DynamicDimensions>(1));
  ASSERT_EQ(ts.first(2), TensorShape<DynamicDimensions>(1, 2));
  ASSERT_EQ(ts.first(3), TensorShape<DynamicDimensions>(1, 2, 3));
  ASSERT_EQ(ts.first(4), TensorShape<DynamicDimensions>(1, 2, 3, 4));
  ASSERT_EQ(ts.first(5), TensorShape<DynamicDimensions>(1, 2, 3, 4, 5));
}

TEST(TensorShapeTest, LastDynamic) {
  TensorShape<DynamicDimensions> ts(1, 2, 3, 4, 5);
  ASSERT_EQ(ts.last(0), TensorShape<DynamicDimensions>());
  ASSERT_EQ(ts.last(1), TensorShape<DynamicDimensions>(5));
  ASSERT_EQ(ts.last(2), TensorShape<DynamicDimensions>(4, 5));
  ASSERT_EQ(ts.last(3), TensorShape<DynamicDimensions>(3, 4, 5));
  ASSERT_EQ(ts.last(4), TensorShape<DynamicDimensions>(2, 3, 4, 5));
  ASSERT_EQ(ts.last(5), TensorShape<DynamicDimensions>(1, 2, 3, 4, 5));
}

TEST(TensorShapeTest, Concatenation) {
  ASSERT_EQ(TensorShape<0>() + TensorShape<0>(), TensorShape<0>());
  ASSERT_EQ(TensorShape<1>(1) + TensorShape<0>(), TensorShape<1>(1));
  ASSERT_EQ(TensorShape<0>() + TensorShape<1>(1), TensorShape<1>(1));
  ASSERT_EQ(TensorShape<2>(1, 2) + TensorShape<3>(1, 2, 3), TensorShape<5>(1, 2, 1, 2, 3));

  
  
  ASSERT_EQ(TensorShape<DynamicDimensions>() + TensorShape<DynamicDimensions>(), TensorShape<DynamicDimensions>());
  ASSERT_EQ(TensorShape<DynamicDimensions>(1) + TensorShape<DynamicDimensions>(), TensorShape<DynamicDimensions>(1));
  ASSERT_EQ(TensorShape<DynamicDimensions>() + TensorShape<DynamicDimensions>(1), TensorShape<DynamicDimensions>(1));
  ASSERT_EQ(TensorShape<DynamicDimensions>(1, 2) + TensorShape<DynamicDimensions>(1, 2, 3), TensorShape<DynamicDimensions>(1, 2, 1, 2, 3));
}

TEST(TensorViewTest, Conversions) {
  TensorView<EmptyBackendTag, int, 4> empty_static_dim{};
  ASSERT_EQ(empty_static_dim.data, nullptr);
  ASSERT_EQ(empty_static_dim.shape, TensorShape<4>());

  TensorView<EmptyBackendTag, int, 4> static_dim{static_cast<int*>(nullptr), {1, 2, 3, 4}};
  // Allowed conversions
  TensorView<EmptyBackendTag, int, DynamicDimensions> dynamic_dim{static_dim};
  ASSERT_EQ(dynamic_dim.shape, static_dim.shape);
  TensorView<EmptyBackendTag, int, 4> static_dim_2(dynamic_dim.to_static_ndim<4>());
  ASSERT_EQ(static_dim_2.shape, static_dim.shape);
  ASSERT_EQ(static_dim_2.shape, dynamic_dim.shape);
}

TEST(VolumeTest, Result) {
  //todo
}

TEST(FlattenTest, Result) {
  //todo
}

TEST(CalculateOffsetsTest, Result) {
  //todo
}

TEST(TensorTest, WontCompile) {
  // TensorShape<5> static_shape_less(1, 2, 3, 4);
  // TensorShape<5> static_shape_more(1, 2, 3, 4, 5, 6);
  // TensorShape<DynamicDimensions>().to_static_ndim<DynamicDimensions>();
  // TensorShape<5>{TensorShape<DynamicDimensions>()};

  // TensorView<EmptyBackendTag, int8_t, 4>(static_cast<int*>(nullptr), {1, 2, 3, 4});
  // TensorView<EmptyBackendTag, int, 4>{TensorView<EmptyBackendTag, int, DynamicDimensions>{}};
  // TensorView<EmptyBackendTag, int, DynamicDimensions>{TensorView<EmptyBackendTag, int8_t, 4>{}};


}

}  // namespace tensor
