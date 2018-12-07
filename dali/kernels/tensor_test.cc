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
  constexpr int test_dim = 5;
  std::array<int64_t, test_dim> test_shape = {1, 2, 3, 4, 5};
  TensorShape<test_dim> zero_tensor;
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(zero_tensor[i], int64_t(0));
  }

  TensorShape<test_dim> a(test_shape);
  TensorShape<test_dim> b(test_shape[0], test_shape[1], test_shape[2], test_shape[3], test_shape[4]);
  ASSERT_EQ(a.size(), test_dim);
  ASSERT_EQ(b.size(), test_dim);
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(a[i], test_shape[i]);
    ASSERT_EQ(b[i], test_shape[i]);
  }

  TensorShape<test_dim> check_construct(a);
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(check_construct[i], a[i]);
  }

  TensorShape<test_dim> check_assign;
  check_assign = a;
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(check_assign[i], a[i]);
  }

  TensorShape<test_dim> check_move_construct(TensorShape<5>{test_shape});
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(check_move_construct[i], test_shape[i]);
  }

  TensorShape<test_dim> check_move_assign;
  check_move_assign = TensorShape<5>{test_shape};
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(check_move_assign[i], test_shape[i]);
  }

}


TEST(TensorShapeTest, DynamicShapeConstructor) {
  TensorShape<DynamicDimensions> zero_tensor;
  ASSERT_EQ(zero_tensor.size(), 0);


  constexpr int test_dim = 5;
  std::array<int64_t, test_dim> test_shape_arr = {1, 2, 3, 4, 5};
  TensorShape<DynamicDimensions> a(test_shape_arr);
  ASSERT_EQ(a.size(), test_dim);
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(a[i], test_shape_arr[i]);
  }


  std::vector<int64_t> test_shape_vec = {1, 2, 3, 4, 5, 6, 7};
  TensorShape<DynamicDimensions> b(test_shape_vec);
  ASSERT_EQ(b.size(), test_shape_vec.size());
  for (int i = 0; i < test_shape_vec.size(); i++) {
    ASSERT_EQ(b[i], test_shape_vec[i]);
  }


  TensorShape<DynamicDimensions> check_construct(a);
  for (int i = 0; i < test_dim; i++) {
    ASSERT_EQ(check_construct[i], a[i]);
  }


  TensorShape<DynamicDimensions> check_assign;
  check_assign = a;
  ASSERT_EQ(check_assign.size(), a.size());
  for (int i = 0; i < a.size(); i++) {
    ASSERT_EQ(check_assign[i], a[i]);
  }

  check_assign = b;
  ASSERT_EQ(check_assign.size(), b.size());
  for (int i = 0; i < b.size(); i++) {
    ASSERT_EQ(check_assign[i], b[i]);
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

void foo(TensorView<EmptyBackendTag, int, 4> &) {}

TEST(TensorViewTest, Assignement) {
  int tab[20];
  int *p = tab;
  TensorView<EmptyBackendTag, int, 4> int_4{p, {1, 2, 3, 4}};
  TensorView<EmptyBackendTag, int, 3> int_3{p, {1, 2, 3}};

  TensorView<EmptyBackendTag, int, 4> int_4_2{int_4};
  TensorView<EmptyBackendTag, int, -1> int_4_3{int_4};
  TensorView<EmptyBackendTag, int, 4> int_4_4{int_4_3.to_static_ndim<4>()};
  // foo(x);
  // int_4_4 = int_4_3;
  // TensorView<EmptyBackendTag, int, DynamicDimensions> int_dyn;
  // // int_4 = int_dyn;
  // int_dyn = int_4;
}

TEST(TensorListShapeTest, Assignement) {

}

TEST(TensorTest, WontCompile) {
  // TensorShape<5> static_shape(1, 2, 3, 4);
  // TensorShape<5> static_shape(1, 2, 3, 4, 5, 6);

}

}  // namespace tensor
