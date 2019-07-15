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
#include "dali/kernels/imgproc/color_manipulation/brightness_contrast.h"
#include "dali/kernels/test/tensor_test_utils.h"

namespace dali {
namespace kernels {
namespace test {

namespace brightness_contrast {

template <template <typename A, typename B> class Pair, typename T1, typename... T>
using FixedFirstTypePairs = std::tuple<Pair<T1, T>...>;

template <template <typename A, typename B> class Pair, typename A, typename B>
struct AllPairsHelper;

template <template <typename A, typename B> class Pair, typename... A, typename... B>
struct AllPairsHelper<Pair, std::tuple<A...>, std::tuple<B...>> {
  using type = dali::detail::tuple_cat_t<FixedFirstTypePairs<Pair, A, B...>...>;
};

template <template <typename A, typename B> class Pair, typename TupleA, typename TupleB>
using AllPairs = typename AllPairsHelper<Pair, TupleA, TupleB>::type;

template <typename Tuple>
struct TupleToGTest;

template <typename... T>
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
template <int nchannels, class T, class Roi>
cv::Mat_<cv::Vec<T, nchannels>> to_mat(const T *ptr, Roi roi, int rows, int cols) {
  auto roi_w = roi.extent().x;
  auto roi_h = roi.extent().y;
  cv::Mat mat(rows, cols, CV_MAKETYPE(cv::DataDepth<std::remove_const_t<T>>::value, nchannels),
              const_cast<T *>(ptr));
  cv::Rect rect(roi.lo.x, roi.lo.y, roi_w, roi_h);
  cv::Mat_<cv::Vec<T, nchannels>> out_copy;
  mat(rect).copyTo(out_copy);
  assert(out_copy.isContinuous());
  return out_copy;
}


void fill_roi(Box<2, int> &roi) {
  roi = {{1, 2},
         {5, 7}};
}


}  // namespace brightness_contrast


template <class InputOutputTypes>
class BrightnessContrastTest : public ::testing::Test {
 protected:
  BrightnessContrastTest() {
    input_.resize(dali::volume(shape_));
  }


  void SetUp() final {
    std::mt19937_64 rng;
    UniformRandomFill(input_, rng, 0., 10.);
    calc_output<typename InputOutputTypes::out>();
    ref_out_tv_ = make_tensor_cpu(ref_output_.data(), shape_);
  }


  std::vector<typename InputOutputTypes::in> input_;
  std::vector<typename InputOutputTypes::out> ref_output_;
  OutTensorCPU<typename InputOutputTypes::out, 3> ref_out_tv_;
  TensorShape<3> shape_ = {240, 320, 3};
  float brightness_ = 4;
  float contrast_ = 3;
  static constexpr size_t ndims = 3;


  template <typename OutputType>
  std::enable_if_t<std::is_integral<OutputType>::value> calc_output() {
    for (auto in : input_) {
      ref_output_.push_back(std::round(in * contrast_ + brightness_));
    }
  }


  template <typename OutputType>
  std::enable_if_t<!std::is_integral<OutputType>::value> calc_output() {
    for (auto in : input_) {
      ref_output_.push_back(in * contrast_ + brightness_);
    }
  }
};


namespace test_types {

template <class InputType, class OutputType>
struct InputOutputTypes {
  using in = InputType;
  using out = OutputType;
};

using ArgTypes = std::tuple<uint8_t, int8_t, uint16_t, int16_t, int32_t, float>;
using MyTypesTuple = brightness_contrast::AllPairs<InputOutputTypes, ArgTypes, ArgTypes>;
using GTestTypes = typename brightness_contrast::TupleToGTest<MyTypesTuple>::type;

}  // namespace test_types


TYPED_TEST_SUITE(BrightnessContrastTest, test_types::GTestTypes);

TYPED_TEST(BrightnessContrastTest, check_kernel) {
  BrightnessContrastCPU<typename TypeParam::in, typename TypeParam::out> kernel;
  check_kernel<decltype(kernel)>();
}


TYPED_TEST(BrightnessContrastTest, SetupTestAndCheckKernel) {
  BrightnessContrastCPU<typename TypeParam::in, typename TypeParam::out> kernel;
  constexpr auto ndims = std::remove_reference_t<decltype(*this)>::ndims;
  KernelContext ctx;
  InTensorCPU<typename TypeParam::in, this->ndims> in(this->input_.data(), this->shape_);
  auto reqs = kernel.Setup(ctx, in, this->brightness_, this->contrast_);
  auto sh = reqs.output_shapes[0][0];
  ASSERT_EQ(this->shape_, sh);
}


TYPED_TEST(BrightnessContrastTest, RunTest) {
  BrightnessContrastCPU<typename TypeParam::in, typename TypeParam::out> kernel;
  constexpr auto ndims = std::remove_reference_t<decltype(*this)>::ndims;
  KernelContext ctx;
  InTensorCPU<typename TypeParam::in, ndims> in(this->input_.data(), this->shape_);
  auto reqs = kernel.Setup(ctx, in, this->brightness_, this->contrast_);
  auto out_shape = reqs.output_shapes[0][0];
  vector<typename TypeParam::out> output;
  output.resize(dali::volume(out_shape));
  OutTensorCPU<typename TypeParam::out, ndims> out(output.data(),
                                                   out_shape.template to_static<ndims>());

  kernel.Run(ctx, out, in, this->brightness_, this->contrast_);
  for (int i = 0; i < out.num_elements(); i++) {
    EXPECT_FLOAT_EQ(this->ref_out_tv_.data[i], out.data[i]) << "Failed at idx: " << i;
  }
}


TYPED_TEST(BrightnessContrastTest, RunTestWithRoi) {
  BrightnessContrastCPU<typename TypeParam::in, typename TypeParam::out> kernel;
  constexpr auto ndims = std::remove_reference_t<decltype(*this)>::ndims;
  KernelContext ctx;
  InTensorCPU<typename TypeParam::in, ndims> in(this->input_.data(), this->shape_);

  typename decltype(kernel)::Roi roi;
  brightness_contrast::fill_roi(roi);

  auto reqs = kernel.Setup(ctx, in, this->brightness_, this->contrast_, &roi);
  auto out_shape = reqs.output_shapes[0][0];
  vector<typename TypeParam::out> output;
  output.resize(dali::volume(out_shape));
  OutTensorCPU<typename TypeParam::out, ndims> out(output.data(),
                                                   out_shape.template to_static<ndims>());

  kernel.Run(ctx, out, in, this->brightness_, this->contrast_, &roi);

  auto mat = brightness_contrast::to_mat<ndims>(this->ref_output_.data(), roi,
                                                this->shape_[0], this->shape_[1]);
  ASSERT_EQ(mat.rows * mat.cols * mat.channels(), out.num_elements())
                        << "Number of elements doesn't match";
  auto ptr = reinterpret_cast<typename TypeParam::out *>(mat.data);
  for (int i = 0; i < out.num_elements(); i++) {
    EXPECT_FLOAT_EQ(ptr[i], out.data[i]) << "Failed at idx: " << i;
  }
}


TYPED_TEST(BrightnessContrastTest, roi_shape) {
  {
    Box<2, int> box{0, 3};
    auto sh = ::dali::kernels::brightness_contrast::roi_shape(box, 3);
    TensorShape<3> ref_sh = {3, 3, 3};
    ASSERT_EQ(ref_sh, sh);
  }
  {
    Box<2, int> box{{0, 2},
                    {5, 6}};
    auto sh = ::dali::kernels::brightness_contrast::roi_shape(box, 666);
    TensorShape<3> ref_sh = {4, 5, 666};
    ASSERT_EQ(ref_sh, sh);
  }
  {
    Box<2, int> box{{0, 0},
                    {0, 0}};
    auto sh = ::dali::kernels::brightness_contrast::roi_shape(box, 666);
    TensorShape<3> ref_sh = {0, 0, 666};
    ASSERT_EQ(ref_sh, sh);
  }
}


}  // namespace test
}  // namespace kernels
}  // namespace dali
