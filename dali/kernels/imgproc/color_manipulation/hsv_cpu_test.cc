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
#include "dali/test/tensor_test_utils.h"
#include "dali/kernels/test/kernel_test_utils.h"
#include "dali/kernels/imgproc/color_manipulation/hsv_cpu.h"
#include "dali/test/cv_mat_utils.h"

namespace dali {
namespace kernels {
namespace hsv {
namespace test {

namespace {

static constexpr int ndims = 3;


void fill_roi(Box<2, int> &roi) {
  roi = {{1, 2},
         {5, 7}};
}


template <class GtestTypeParam>
using HsvKernel = HsvCpu<typename GtestTypeParam::Out, typename GtestTypeParam::In>;

}  // namespace


template <class InputOutputTypes>
class HsvCpuTest : public ::testing::Test {
  using In = typename InputOutputTypes::In;
  using Out = typename InputOutputTypes::Out;

 protected:
  HsvCpuTest() {
    input_.resize(dali::volume(shape_));
    ref_output_.resize(dali::volume(shape_));
  }


  void SetUp() final {
    std::mt19937_64 rng;
    UniformRandomFill(input_, rng, 0., 10.);
    calc_output<Out>();
    ref_out_tv_ = make_tensor_cpu(ref_output_.data(), shape_);
  }


  std::vector<In> input_;
  std::vector<Out> ref_output_;
  OutTensorCPU<Out, ndims> ref_out_tv_;
  TensorShape<3> shape_ = {23, 45, ndims};
  float hue_ = 2.8f;
  float saturation_ = 1.9f;
  float value_ = 0.3f;


  template <typename OutputType>
  std::enable_if_t<std::is_integral<OutputType>::value> calc_output() {
    for (size_t i = 0; i < input_.size(); i += ndims) {
      ref_output_[i + 0] = std::round(input_[i + 0] + hue_);
      ref_output_[i + 1] = std::round(input_[i + 1] * saturation_);
      ref_output_[i + 2] = std::round(input_[i + 2] * value_);
    }
  }


  template <typename OutputType>
  std::enable_if_t<!std::is_integral<OutputType>::value> calc_output() {
    for (size_t i = 0; i < input_.size(); i += ndims) {
      ref_output_[i + 0] = input_[i + 0] + hue_;
      ref_output_[i + 1] = input_[i + 1] * saturation_;
      ref_output_[i + 2] = input_[i + 2] * value_;
    }
  }
};


using TestTypes = std::tuple<uint8_t, int8_t, uint16_t, int16_t, int32_t, float>;
INPUT_OUTPUT_TYPED_TEST_SUITE(HsvCpuTest, TestTypes);


TYPED_TEST(HsvCpuTest, check_kernel) {
  HsvKernel<TypeParam> kernel;
  check_kernel<decltype(kernel)>();
}


TYPED_TEST(HsvCpuTest, SetupTestAndCheckKernel) {
  HsvKernel<TypeParam> kernel;
  KernelContext ctx;
  InTensorCPU<typename TypeParam::In, ndims> in(this->input_.data(), this->shape_);
  auto reqs = kernel.Setup(ctx, in, this->hue_, this->saturation_, this->value_);
  auto sh = reqs.output_shapes[0][0];
  ASSERT_EQ(this->shape_, sh);
}


TYPED_TEST(HsvCpuTest, RunTest) {
  HsvKernel<TypeParam> kernel;
  KernelContext ctx;
  InTensorCPU<typename TypeParam::In, ndims> in(this->input_.data(), this->shape_);
  auto reqs = kernel.Setup(ctx, in, this->hue_, this->saturation_, this->value_);
  auto out_shape = reqs.output_shapes[0][0];
  vector<typename TypeParam::Out> output;
  output.resize(dali::volume(out_shape));
  OutTensorCPU<typename TypeParam::Out, ndims> out(output.data(),
                                                   out_shape.template to_static<ndims>());

  kernel.Run(ctx, out, in, this->hue_, this->saturation_, this->value_);
  for (int i = 0; i < out.num_elements(); i++) {
    EXPECT_FLOAT_EQ(this->ref_out_tv_.data[i], out.data[i]) << "Failed at idx: " << i;
  }
}


TYPED_TEST(HsvCpuTest, RunTestWithRoi) {
  HsvKernel<TypeParam> kernel;
  KernelContext ctx;
  InTensorCPU<typename TypeParam::In, ndims> in(this->input_.data(), this->shape_);

  typename decltype(kernel)::Roi roi;
  fill_roi(roi);

  auto reqs = kernel.Setup(ctx, in, this->hue_, this->saturation_, this->value_, &roi);
  auto out_shape = reqs.output_shapes[0][0];
  vector<typename TypeParam::Out> output;
  output.resize(dali::volume(out_shape));
  OutTensorCPU<typename TypeParam::Out, ndims> out(output.data(),
                                                   out_shape.template to_static<ndims>());

  kernel.Run(ctx, out, in, this->hue_, this->saturation_, this->value_, &roi);

  auto mat = testing::copy_to_mat<ndims>(
    roi, this->ref_output_.data(), this->shape_[0], this->shape_[1]);

  ASSERT_EQ(mat.rows * mat.cols * mat.channels(), out.num_elements())
                        << "Number of elements doesn't match";
  auto ptr = reinterpret_cast<typename TypeParam::Out *>(mat.data);
  for (int i = 0; i < out.num_elements(); i++) {
    EXPECT_FLOAT_EQ(ptr[i], out.data[i]) << "Failed at idx: " << i;
  }
}


TYPED_TEST(HsvCpuTest, roi_shape) {
  {
    Box<2, int> box{0, 3};
    auto sh = ::dali::kernels::ShapeFromRoi(box, 3);
    TensorShape<3> ref_sh = {3, 3, 3};
    ASSERT_EQ(ref_sh, sh);
  }
  {
    Box<2, int> box{{0, 2},
                    {5, 6}};
    auto sh = ::dali::kernels::ShapeFromRoi(box, 666);
    TensorShape<3> ref_sh = {4, 5, 666};
    ASSERT_EQ(ref_sh, sh);
  }
  {
    Box<2, int> box{{0, 0},
                    {0, 0}};
    auto sh = ::dali::kernels::ShapeFromRoi(box, 666);
    TensorShape<3> ref_sh = {0, 0, 666};
    ASSERT_EQ(ref_sh, sh);
  }
}


}  // namespace test
}  // namespace hsv
}  // namespace kernels
}  // namespace dali
