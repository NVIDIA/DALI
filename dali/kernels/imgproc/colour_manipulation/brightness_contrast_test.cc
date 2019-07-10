// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include <tuple>
#include "dali/kernels/imgproc/colour_manipulation/brightness_contrast.h"
#include "dali/kernels/test/tensor_test_utils.h"

namespace dali {
namespace kernels {
namespace test {

using brightness_contrast::Roi;

namespace detail {

template<template<typename A, typename B> class Pair, typename T1, typename... T>
using FixedFirstTypePairs = std::tuple<Pair<T1, T>...>;

template<template<typename A, typename B> class Pair, typename A, typename B>
struct AllPairsHelper;

template<template<typename A, typename B> class Pair, typename... A, typename... B>
struct AllPairsHelper<Pair, std::tuple<A...>, std::tuple<B...>> {
  using type = dali::detail::tuple_cat_t<FixedFirstTypePairs<Pair, A, B...>...>;
};

template<template<typename A, typename B> class Pair, typename TupleA, typename TupleB>
using AllPairs = typename AllPairsHelper<Pair, TupleA, TupleB>::type;

template<typename Tuple>
struct TupleToGTest;

template<typename... T>
struct TupleToGTest<std::tuple<T...>> {
  using type = testing::Types<T...>;
};


/**
 * Creates cv::Mat based on provided arguments.
 * This mat is for roi-testing purposes only. Particularly, it doesn't care
 * about image type (i.e. number of channels), so don't try to imshow it.
 *
 * @param ptr Can't be const, due to cv::Mat API
 * @param rows height of the input image
 * @param cols width of the input image
 */
template<int nchannels, class T>
cv::Mat_<T> to_mat(T *ptr, Roi roi, int rows, int cols) {
  cv::Mat_<T> mat(rows, cols * nchannels, ptr);
  cv::Rect rect(roi.x * nchannels, roi.y, roi.w * nchannels, roi.h);
  auto roimat = cv::Mat_<T>(mat, rect);
  roimat = roimat.clone();  // Make cv::Mat continuous
  return roimat;
}

}  // namespace detail


template<class InputOutputTypes>
class BrightnessContrastTest : public ::testing::Test {
 protected:
  BrightnessContrastTest() {
    input_.resize(dali::volume(shape_));
  }


  void SetUp() final {
    std::mt19937_64 rng;
    UniformRandomFill(input_, rng, 0., 10.);
    calc_output();
  }


  std::vector<typename InputOutputTypes::in> input_;
  std::vector<typename InputOutputTypes::out> ref_output_;
  TensorShape<3> shape_ = {240, 320, 3};
  typename InputOutputTypes::in brightness_ = 4;
  typename InputOutputTypes::in contrast_ = 3;
  static constexpr size_t ndims = 3;


  void calc_output() {
    for (auto in : input_) {
      ref_output_.push_back(in * contrast_ + brightness_);
    }
  }
};


namespace test_types {

template<class InputType, class OutputType>
struct InputOutputTypes {
  using in = InputType;
  using out = OutputType;
};

using ArgTypes = std::tuple<uint8_t, int8_t, uint16_t, int16_t, int32_t, float>;
using MyTypesTuple = detail::AllPairs<InputOutputTypes, ArgTypes, ArgTypes>;
using GTestTypes = typename detail::TupleToGTest<MyTypesTuple>::type;

}  // namespace test_types


TYPED_TEST_SUITE(BrightnessContrastTest, test_types::GTestTypes);

TYPED_TEST(BrightnessContrastTest, check_kernel) {
  BrightnessContrast<kernels::ComputeCPU, typename TypeParam::in, typename TypeParam::out> kernel;
  check_kernel<decltype(kernel)>();
}


TYPED_TEST(BrightnessContrastTest, SetupTestAndCheckKernel) {
  BrightnessContrast<kernels::ComputeCPU, typename TypeParam::in, typename TypeParam::out> kernel;
  KernelContext ctx;
  InTensorCPU<typename TypeParam::in, this->ndims> in(this->input_.data(), this->shape_);
  auto reqs = kernel.Setup(ctx, in, this->brightness_, this->contrast_);
  auto sh = reqs.output_shapes[0][0];
  ASSERT_EQ(this->shape_, sh);
}


TYPED_TEST(BrightnessContrastTest, RunTest) {
  BrightnessContrast<kernels::ComputeCPU, typename TypeParam::in, typename TypeParam::out> kernel;
  KernelContext ctx;
  InTensorCPU<typename TypeParam::in, 3> in(this->input_.data(), this->shape_);
  auto reqs = kernel.Setup(ctx, in, this->brightness_, this->contrast_);
  auto out_shape = reqs.output_shapes[0][0];
  vector<typename TypeParam::out> output;
  output.resize(dali::volume(out_shape));
  OutTensorCPU<typename TypeParam::out, this->ndims> out(
          output.data(), out_shape.template to_static<this->ndims>());

  kernel.Run(ctx, out, in, this->brightness_, this->contrast_);

  ASSERT_EQ(this->ref_output_.size(), out.num_elements()) << "Number of elements doesn't match";
  for (int i = 0; i < out.num_elements(); i++) {
    EXPECT_EQ(this->ref_output_[i], out.data[i]) << "Failed at idx: " << i;
  }
}


TYPED_TEST(BrightnessContrastTest, RunTestWithRoi) {
  BrightnessContrast<kernels::ComputeCPU, typename TypeParam::in, typename TypeParam::out> kernel;
  KernelContext ctx;
  InTensorCPU<typename TypeParam::in, this->ndims> in(this->input_.data(), this->shape_);

  Roi roi = {1, 1, 2, 2};

  auto reqs = kernel.Setup(ctx, in, this->brightness_, this->contrast_, roi);
  auto out_shape = reqs.output_shapes[0][0];
  vector<typename TypeParam::out> output;
  output.resize(dali::volume(out_shape));
  OutTensorCPU<typename TypeParam::out, this->ndims> out(
          output.data(), out_shape.template to_static<this->ndims>());

  kernel.Run(ctx, out, in, this->brightness_, this->contrast_, roi);

  auto mat = detail::to_mat<this->ndims>(this->ref_output_.data(), roi, this->shape_[0],
                                         this->shape_[1]);
  ASSERT_EQ(mat.rows * mat.cols, out.num_elements()) << "Number of elements doesn't match";
  auto ptr = reinterpret_cast<typename TypeParam::out *>(mat.data);
  for (int i = 0; i < out.num_elements(); i++) {
    EXPECT_EQ(ptr[i], out.data[i]) << "Failed at idx: " << i;
  }
}


}  // namespace test
}  // namespace kernels
}  // namespace dali
