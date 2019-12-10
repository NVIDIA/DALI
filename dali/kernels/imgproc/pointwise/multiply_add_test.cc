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
#include "dali/kernels/imgproc/pointwise/multiply_add.h"
#include "dali/test/cv_mat_utils.h"

namespace dali {
namespace kernels {
namespace multiply_add {
namespace test {

namespace {

void fill_roi(Box<2, int> &roi) {
  roi = {{1, 2},
         {5, 7}};
}

}  // namespace

template <class InputOutputTypes>
class MultiplyAddCpuTest : public ::testing::Test {
 protected:
  MultiplyAddCpuTest() {
    input_.resize(dali::volume(shape_));
  }


  void SetUp() final {
    std::mt19937_64 rng;
    UniformRandomFill(input_, rng, 0., 10.);
    calc_output<typename InputOutputTypes::Out>();
    ref_out_tv_ = make_tensor_cpu(ref_output_.data(), shape_);
  }


  std::vector<typename InputOutputTypes::In> input_;
  std::vector<typename InputOutputTypes::Out> ref_output_;
  OutTensorCPU<typename InputOutputTypes::Out, 3> ref_out_tv_;
  TensorShape<3> shape_ = {240, 320, 3};
  float addend_ = 4;
  float multiplier_ = 3;
  static constexpr size_t ndims = 3;


  template <typename OutputType>
  std::enable_if_t<std::is_integral<OutputType>::value> calc_output() {
    for (auto in : input_) {
      ref_output_.push_back(std::round(in * multiplier_ + addend_));
    }
  }


  template <typename OutputType>
  std::enable_if_t<!std::is_integral<OutputType>::value> calc_output() {
    for (auto in : input_) {
      ref_output_.push_back(in * multiplier_ + addend_);
    }
  }
};


using TestTypes = std::tuple<uint8_t, int16_t, int32_t, float>;
INPUT_OUTPUT_TYPED_TEST_SUITE(MultiplyAddCpuTest, TestTypes);

namespace {

template <class GtestTypeParam>
using MultiplyAddKernel = MultiplyAddCpu
        <typename GtestTypeParam::Out, typename GtestTypeParam::In>;

}  // namespace



TYPED_TEST(MultiplyAddCpuTest, check_kernel) {
  MultiplyAddKernel<TypeParam> kernel;
  check_kernel<decltype(kernel)>();
}


TYPED_TEST(MultiplyAddCpuTest, SetupTestAndCheckKernel) {
  MultiplyAddKernel<TypeParam> kernel;
  constexpr auto ndims = std::remove_reference_t<decltype(*this)>::ndims;
  KernelContext ctx;
  InTensorCPU<typename TypeParam::In, ndims> in(this->input_.data(), this->shape_);
  auto reqs = kernel.Setup(ctx, in, this->addend_, this->multiplier_);
  auto sh = reqs.output_shapes[0][0];
  ASSERT_EQ(this->shape_, sh);
}


TYPED_TEST(MultiplyAddCpuTest, RunTest) {
  MultiplyAddKernel<TypeParam> kernel;
  constexpr auto ndims = std::remove_reference_t<decltype(*this)>::ndims;
  KernelContext ctx;
  InTensorCPU<typename TypeParam::In, ndims> in(this->input_.data(), this->shape_);
  auto reqs = kernel.Setup(ctx, in, this->addend_, this->multiplier_);
  auto out_shape = reqs.output_shapes[0][0];
  vector<typename TypeParam::Out> output;
  output.resize(dali::volume(out_shape));
  OutTensorCPU<typename TypeParam::Out, ndims> out(output.data(),
                                                   out_shape.template to_static<ndims>());

  kernel.Run(ctx, out, in, this->addend_, this->multiplier_);
  for (int i = 0; i < out.num_elements(); i++) {
    EXPECT_FLOAT_EQ(this->ref_out_tv_.data[i], out.data[i]) << "Failed at idx: " << i;
  }
}


TYPED_TEST(MultiplyAddCpuTest, RunTestWithRoi) {
  MultiplyAddKernel<TypeParam> kernel;
  constexpr auto ndims = std::remove_reference_t<decltype(*this)>::ndims;
  KernelContext ctx;
  InTensorCPU<typename TypeParam::In, ndims> in(this->input_.data(), this->shape_);

  typename decltype(kernel)::Roi roi;
  fill_roi(roi);

  auto reqs = kernel.Setup(ctx, in, this->addend_, this->multiplier_, &roi);
  auto out_shape = reqs.output_shapes[0][0];
  vector<typename TypeParam::Out> output;
  output.resize(dali::volume(out_shape));
  OutTensorCPU<typename TypeParam::Out, ndims> out(output.data(),
                                                   out_shape.template to_static<ndims>());

  kernel.Run(ctx, out, in, this->addend_, this->multiplier_, &roi);

  auto mat = testing::copy_to_mat<ndims>(
    roi, this->ref_output_.data(), this->shape_[0], this->shape_[1]);

  ASSERT_EQ(volume(roi.extent()) * ndims, out.num_elements())
                        << "Number of elements doesn't match";
  auto ptr = reinterpret_cast<typename TypeParam::Out *>(mat.data);
  for (int i = 0; i < out.num_elements(); i++) {
    EXPECT_FLOAT_EQ(ptr[i], out.data[i]) << "Failed at idx: " << i;
  }
}




}  // namespace test
}  // namespace multiply_add
}  // namespace kernels
}  // namespace dali
