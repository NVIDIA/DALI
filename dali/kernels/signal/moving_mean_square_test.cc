// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/test/tensor_test_utils.h"
#include "dali/kernels/signal/moving_mean_square.h"

namespace dali {
namespace kernels {
namespace signal {
namespace test {

template<class InputType>
class MovingMeanSquareCpuTest : public ::testing::Test {
 public:
  MovingMeanSquareCpuTest() {
    input_.resize(shape_.num_elements());
  }


  void SetUp() final {
    std::mt19937_64 rng;
    FillInput(rng);
    calc_output();
  }


  std::vector<InputType> input_;
  std::vector<float> ref_output_;
  int window_size_ = 2048;
  int buffer_length_ = 16000;
  int reset_interval_ = 5001;
  TensorShape<1> shape_ = {buffer_length_};
  TensorShape<1> out_shape_ = {buffer_length_};

 private:
  void calc_output() {
    ref_output_.resize(buffer_length_);
    float factor = 1.0f / window_size_;
    for (int i = 0; i < buffer_length_; i++) {
      acc_t<InputType> sum = 0;
      for (int j = std::max(0, i - window_size_ + 1); j <= i; j++) {
        acc_t<InputType> x = input_[j];
        sum += (x * x);
      }
      ref_output_[i] = ConvertSat<float>(factor * sum);
    }
  }


  template<typename RNG, typename T = InputType>
  std::enable_if_t<std::is_signed<T>::value>
  FillInput(RNG &rng) {
    UniformRandomFill(input_, rng, -100, 100);
  }


  template<typename RNG, typename T = InputType>
  std::enable_if_t<!std::is_signed<T>::value>
  FillInput(RNG &rng) {
    UniformRandomFill(input_, rng, 0, 100);
  }
};

using TestTypes = ::testing::Types<uint8_t, uint16_t, int8_t, int16_t, int32_t, float>;
TYPED_TEST_SUITE(MovingMeanSquareCpuTest, TestTypes);

using signal::MovingMeanSquareCpu;

TYPED_TEST(MovingMeanSquareCpuTest, SetupTest) {
  MovingMeanSquareCpu<TypeParam> kernel;
  KernelContext ctx;
  InTensorCPU<TypeParam, 1> in(this->input_.data(), this->shape_);
  auto reqs = kernel.Setup(ctx, in, {this->window_size_});
  ASSERT_EQ(this->out_shape_, reqs.output_shapes[0][0]) << "Kernel::Setup provides incorrect shape";
}


TYPED_TEST(MovingMeanSquareCpuTest, RunTest) {
  MovingMeanSquareCpu<TypeParam> kernel;
  KernelContext ctx;
  InTensorCPU<TypeParam, 1> in(this->input_.data(), this->shape_);

  auto reqs = kernel.Setup(ctx, in, {this->window_size_});

  auto out_shape = reqs.output_shapes[0][0];
  std::vector<float> output;
  output.resize(out_shape.num_elements());
  OutTensorCPU<float, 1> out(output.data(), out_shape.template to_static<1>());

  kernel.Run(ctx, out, in, {this->window_size_, this->reset_interval_});

  auto ref_tv = TensorView<StorageCPU, float>(this->ref_output_.data(), this->out_shape_);
  Check(out, ref_tv, EqualRelative(1e-5));
}


}  // namespace test
}  // namespace signal
}  // namespace kernels
}  // namespace dali
