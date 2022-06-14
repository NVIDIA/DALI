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

#include "dali/core/tensor_view.h"
#include "dali/pipeline/data/sequence_utils.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/views.h"


namespace dali {

template <typename T_, int ndim_, int ndims_to_unfold_, typename Storage_ = StorageCPU>
struct unfold_params {
  using Storage = Storage_;
  using T = T_;
  static constexpr int ndim = ndim_;
  static constexpr int ndims_to_unfold = ndims_to_unfold_;
};

using Types =
    ::testing::Types<unfold_params<int, 3, 0>, unfold_params<int, 3, 1>, unfold_params<int, 3, 2>,
                     unfold_params<int, 3, 3>, unfold_params<int, 1, 0>, unfold_params<int, 1, 1>,
                     unfold_params<int, 0, 0>, unfold_params<uint8_t, 3, 0>,
                     unfold_params<double, 3, 1>, unfold_params<uint8_t, 3, 2>,
                     unfold_params<double, 3, 3>, unfold_params<double, 1, 0>,
                     unfold_params<double, 1, 1>, unfold_params<double, 0, 0>>;

template <typename params>
class UnfoldTensorViewTest : public ::testing::Test {
 protected:
  template <typename View, typename Shape>
  void TestUnfolding(const View& view, const Shape& slice_shape, int num_slices, int slice_stride) {
    auto range = sequence_utils::unfolded_view_range<params::ndims_to_unfold>(view);
    static_assert(
        std::is_same_v<TensorView<typename params::Storage, typename params::T, Shape::static_ndim>,
                       decltype(range[0])>);
    ASSERT_EQ(range.NumSlices(), num_slices);
    ASSERT_EQ(range.SliceShape(), slice_shape);
    ASSERT_EQ(range.SliceSize(), slice_stride);
    int num_iters = 0;
    for (auto&& slice : range) {
      EXPECT_EQ(slice.data, view.data + num_iters * slice_stride);
      EXPECT_EQ(slice.shape, slice_shape);
      num_iters++;
    }
    EXPECT_EQ(num_iters, num_slices);
  }
};

TYPED_TEST_SUITE(UnfoldTensorViewTest, Types);

TYPED_TEST(UnfoldTensorViewTest, UnfoldStatic) {
  using Storage = typename TypeParam::Storage;
  using T = typename TypeParam::T;
  TensorShape<3> max_shape = {7, 3, 5};
  TensorShape<TypeParam::ndim> shape = max_shape.first(TypeParam::ndim);
  T val;
  const TensorView<Storage, T, TypeParam::ndim> view(&val, shape);
  TensorShape<TypeParam::ndim - TypeParam::ndims_to_unfold> slice_shape =
      shape.last(TypeParam::ndim - TypeParam::ndims_to_unfold);
  auto slice_stride = volume(slice_shape);
  auto num_slices = volume(shape.first(TypeParam::ndims_to_unfold));
  this->TestUnfolding(view, slice_shape, num_slices, slice_stride);
}

TYPED_TEST(UnfoldTensorViewTest, UnfoldDynamic) {
  using Storage = typename TypeParam::Storage;
  using T = typename TypeParam::T;
  TensorShape<> max_shape = {3, 7, 5};
  TensorShape<> shape = max_shape.first(TypeParam::ndim);
  T val;
  TensorView<Storage, T, DynamicDimensions> view(&val, shape);
  TensorShape<> slice_shape = shape.last(TypeParam::ndim - TypeParam::ndims_to_unfold);
  auto slice_stride = volume(slice_shape);
  auto num_slices = volume(shape.first(TypeParam::ndims_to_unfold));
  this->TestUnfolding(view, slice_shape, num_slices, slice_stride);
}

TEST(UnfoldTensorViewTest, LeadingZeroVol) {
  float val;
  const TensorView<StorageCPU, float, 4> view(&val, {0, 100, 100, 3});
  auto range_0 = sequence_utils::unfolded_view_range<0>(view);
  ASSERT_EQ(range_0.NumSlices(), 1);
  static_assert(std::is_same<TensorView<StorageCPU, float, 4>, decltype(range_0[0])>::value);
  auto slice = range_0[0];
  EXPECT_EQ(slice.data, view.data);
  EXPECT_EQ(slice.shape, view.shape);
  auto range_1 = sequence_utils::unfolded_view_range<1>(view);
  ASSERT_EQ(range_1.NumSlices(), 0);
  for (auto&& slice : range_1) {
    FAIL() << "The range is empty, there should be zero iterations over it";
  }
}

}  // namespace dali
