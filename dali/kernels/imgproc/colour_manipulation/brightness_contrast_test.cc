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
#include "brightness_contrast_cpu.h"
#include "dali/kernels/test/tensor_test_utils.h"

using namespace std;

namespace dali {
namespace kernels {
namespace test {



// TODO First brightness, then contrast
// TODO Input and Output type: some tuple or sth
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


  std::vector<double> input_;
  std::vector<double> ref_output_;
  TensorShape<3> shape_ = {640, 480, 3};
  double brightness_ = 4;
  double contrast_ = 3;


  void calc_output() {
    for (auto in : input_) {
      ref_output_.push_back(in * contrast_ + brightness_);
    }
  }
};

//using MyTypes = ::testing::Types<double>;
//TYPED_TEST_SUITE(BrightnessContrastTest, MyTypes);

TEST_F(BrightnessContrastTest, SetupTest) {
  BrightnessContrastCPU<kernels::ComputeCPU, double, double> kernel;
  KernelContext ctx;
  InTensorCPU<double, 3> in(this->input_.data(), this->shape_);
  auto reqs = kernel.Setup(ctx, in);
  auto sh = reqs.output_shapes[0][0];
  ASSERT_EQ(this->shape_, sh);
}


TEST_F(BrightnessContrastTest, RunTest) {
  BrightnessContrastCPU<kernels::ComputeCPU, double, double> kernel;
  KernelContext ctx;
  InTensorCPU<double, 3> in(this->input_.data(), this->shape_);
  vector<double> output;
  output.resize(dali::volume(this->shape_));
  OutTensorCPU<double, 3> out(output.data(), this->shape_);
  kernel.Run(ctx, in, out, this->brightness_, this->contrast_);
  ASSERT_EQ(this->ref_output_.size(), out.num_elements()) << "Number of element doesn't match";
  for (int i = 0; i < out.num_elements(); i++) {
    EXPECT_EQ(this->ref_output_[i], out.data[i]);
  }
}


}
}
}