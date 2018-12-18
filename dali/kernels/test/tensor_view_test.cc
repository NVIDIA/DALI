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

#include "dali/kernels/tensor_shape.h"
#include "dali/kernels/tensor_view.h"

namespace dali {
namespace kernels {

TEST(ShapeDimTest, Value) {
  int x1[] = {1, 2, 3};
  EXPECT_EQ(ShapeDim(x1), 3);
  std::array<int, 5> x2;
  EXPECT_EQ(ShapeDim(x2), 5);
  std::vector<int> x3 = {1, 2, 3, 4};
  EXPECT_EQ(ShapeDim(x3), 4);
}

TEST(TensorViewTest, StaticConstructors) {
  TensorView<EmptyBackendTag, int, 4> empty_static_dim{};
  EXPECT_EQ(empty_static_dim.data, nullptr);
  EXPECT_EQ(empty_static_dim.shape, TensorShape<4>());
}

TEST(TensorViewTest, Conversions) {
  TensorView<EmptyBackendTag, int, 4> static_dim{static_cast<int*>(nullptr), {1, 2, 3, 4}};
  ASSERT_EQ(static_dim.dim(), 4);
  // Allowed conversions
  TensorView<EmptyBackendTag, int, DynamicDimensions> dynamic_dim{static_dim};
  EXPECT_EQ(dynamic_dim.shape, static_dim.shape);
  ASSERT_EQ(dynamic_dim.dim(), 4);
  TensorView<EmptyBackendTag, int, 4> static_dim_2(dynamic_dim.to_static<4>());
  EXPECT_EQ(static_dim_2.shape, static_dim.shape);
  EXPECT_EQ(static_dim_2.shape, dynamic_dim.shape);

  dynamic_dim = TensorView<EmptyBackendTag, int, 2>{static_cast<int*>(nullptr), {1, 2}};
  ASSERT_EQ(dynamic_dim.dim(), 2);
}

TEST(TensorViewTest, Addressing) {
  TensorView<EmptyBackendTag, int, 3> tv{static_cast<int*>(nullptr), {4, 100, 50}};
  EXPECT_EQ(tv(0, 0, 0), static_cast<int*>(nullptr));
  EXPECT_EQ(tv(0, 0, 1), static_cast<int*>(nullptr) + 1);
  EXPECT_EQ(tv(0, 1, 0), static_cast<int*>(nullptr) + 50);
  EXPECT_EQ(tv(1, 0, 0), static_cast<int*>(nullptr) + 5000);
  EXPECT_EQ(tv(1, 1, 1), static_cast<int*>(nullptr) + 5051);
  EXPECT_EQ(tv(1, 1), static_cast<int*>(nullptr) + 5050);
  // EXPECT_EQ(tv(1), static_cast<int*>(nullptr) + 5000); // TODO - this is ambigous
}

TEST(TensorListViewTest, Constructor) {
  TensorListView<EmptyBackendTag, int, 3> tlv{
      static_cast<int*>(nullptr), {{4, 100, 50}, {2, 10, 5}, {4, 50, 25}, {4, 100, 50}}};
  ASSERT_EQ(tlv.size(), 4);
  ASSERT_EQ(tlv.sample_dim(), 3);
  TensorListView<EmptyBackendTag, int> tlv_dynamic(tlv);
  ASSERT_EQ(tlv_dynamic.size(), 4);
  ASSERT_EQ(tlv_dynamic.sample_dim(), 3);
}

TEST(TensorListViewTest, OperatorSubscript) {
  TensorListView<EmptyBackendTag, int, 3> tlv{
      static_cast<int*>(nullptr), {{4, 100, 50}, {2, 10, 5}, {4, 50, 25}, {4, 100, 50}}};
  EXPECT_EQ(tlv[0].shape.size(), 3);
  EXPECT_EQ(tlv[0].data, static_cast<int*>(nullptr));
  EXPECT_EQ(tlv[1].data, static_cast<int*>(nullptr) + 4 * 100 * 50);
  EXPECT_EQ(tlv[2].data, static_cast<int*>(nullptr) + 4 * 100 * 50 + 2 * 10 * 5);
  EXPECT_EQ(tlv[3].data, static_cast<int*>(nullptr) + 4 * 100 * 50 + 2 * 10 * 5 + 4 * 50 * 25);
}

TEST(TensorListViewTest, ObtainingTensorViewFromStatic) {
  TensorListView<EmptyBackendTag, int, 3> tlv_static{
      static_cast<int*>(nullptr), {{4, 100, 50}, {2, 10, 5}, {4, 50, 25}, {4, 100, 50}}};

  auto t0 = tlv_static[0];
  static_assert(std::is_same<decltype(t0), TensorView<EmptyBackendTag, int, 3>>::value, "Wrong type");
  auto t1 = tlv_static.tensor_view<3>(1);
  static_assert(std::is_same<decltype(t1), TensorView<EmptyBackendTag, int, 3>>::value, "Wrong type");
  auto t2 = tlv_static.tensor_view<DynamicDimensions>(2);
  static_assert(std::is_same<decltype(t2), TensorView<EmptyBackendTag, int, DynamicDimensions>>::value, "Wrong type");
  EXPECT_EQ(t2.dim(), 3);
}

TEST(TensorListViewTest, ObtainingTensorViewFromDynamic) {
  TensorListView<EmptyBackendTag, int> tlv_dynamic{
      static_cast<int*>(nullptr), {{4, 100, 50}, {2, 10, 5}, {4, 50, 25}, {4, 100, 50}}};
  auto t0 = tlv_dynamic[0];
  static_assert(std::is_same<decltype(t0), TensorView<EmptyBackendTag, int, DynamicDimensions>>::value, "Wrong type");
  auto t1 = tlv_dynamic.tensor_view<3>(1);
  static_assert(std::is_same<decltype(t1), TensorView<EmptyBackendTag, int, 3>>::value, "Wrong type");
  auto t2 = tlv_dynamic.tensor_view<DynamicDimensions>(2);
  static_assert(std::is_same<decltype(t2), TensorView<EmptyBackendTag, int, DynamicDimensions>>::value, "Wrong type");
  EXPECT_EQ(t2.dim(), 3);
}

}  // namespace kernels
}  // namespace dali
