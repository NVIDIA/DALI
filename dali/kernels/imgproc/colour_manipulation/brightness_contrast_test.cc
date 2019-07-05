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
#include "brightness_contrast.h"
#include "dali/kernels/test/tensor_test_utils.h"

using namespace std;

namespace dali {
namespace kernels {
namespace test {


// TODO First brightness, then contrast
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
  TensorShape<3> shape_ = {480, 640, 3}; // TODO parameterize
  typename InputOutputTypes::in brightness_ = 4;
  typename InputOutputTypes::in contrast_ = 3;


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

//TODO remaining types
using t1 = InputOutputTypes<double, double>;
using t2 = InputOutputTypes<int, int>;
using t3 = InputOutputTypes<float, float>;
using t4 = InputOutputTypes<int, float>;
using MyTypes = ::testing::Types<t1, t2, t3, t4>;
}  // namespace test_types


TYPED_TEST_SUITE(BrightnessContrastTest, test_types::MyTypes);


TYPED_TEST(BrightnessContrastTest, SetupTest) {
  BrightnessContrast<kernels::ComputeCPU, typename TypeParam::in, typename TypeParam::out> kernel;
  KernelContext ctx;
  InTensorCPU<typename TypeParam::in, 3> in(this->input_.data(), this->shape_);
  auto reqs = kernel.Setup(ctx, in);
  auto sh = reqs.output_shapes[0][0];
  ASSERT_EQ(this->shape_, sh);
}


TYPED_TEST(BrightnessContrastTest, RunTest) {
  BrightnessContrast<kernels::ComputeCPU, typename TypeParam::in, typename TypeParam::out> kernel;
  KernelContext ctx;
  InTensorCPU<typename TypeParam::in, 3> in(this->input_.data(), this->shape_);
  auto reqs = kernel.Setup(ctx, in);
  auto out_shape = reqs.output_shapes[0][0];
  vector<typename TypeParam::out> output;
  output.resize(dali::volume(out_shape));
  OutTensorCPU<typename TypeParam::out, 3> out(output.data(), out_shape.template to_static<3>());

  kernel.Run(ctx, in, out, this->brightness_, this->contrast_);

  ASSERT_EQ(this->ref_output_.size(), out.num_elements()) << "Number of element doesn't match";
  for (int i = 0; i < out.num_elements(); i++) {
    EXPECT_EQ(this->ref_output_[i], out.data[i]);
  }
}



}  // namespace test
}  // namespace kernels
}  // namespace dali