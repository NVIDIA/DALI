// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
namespace test {

namespace {
const int kNDims = 1;
}


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
  TensorShape<kNDims> shape_ = {buffer_length_};
  TensorShape<kNDims> out_shape_ = {buffer_length_ - window_size_ + 1};

 private:
  void calc_output() {
    ref_output_.resize(buffer_length_ - window_size_ + 1);
    for (int i = 0; i < buffer_length_ - window_size_ + 1; i++) {
      float sumsq = 0;
      for (int j = 0; j < window_size_; j++) {
        auto val = static_cast<float>(input_[i + j]);
        sumsq += val * val;
      }
      ref_output_[i] = sumsq / window_size_;
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
  InTensorCPU<TypeParam, kNDims> in(this->input_.data(), this->shape_);
  auto reqs = kernel.Setup(ctx, in, {this->window_size_});
  ASSERT_EQ(this->out_shape_, reqs.output_shapes[0][0]) << "Kernel::Setup provides incorrect shape";
}


TYPED_TEST(MovingMeanSquareCpuTest, RunTest) {
  MovingMeanSquareCpu<TypeParam> kernel;
  KernelContext ctx;
  InTensorCPU<TypeParam, kNDims> in(this->input_.data(), this->shape_);

  auto reqs = kernel.Setup(ctx, in, {this->window_size_});

  auto out_shape = reqs.output_shapes[0][0];
  std::vector<float> output;
  output.resize(out_shape.num_elements());
  OutTensorCPU<float, kNDims> out(output.data(), out_shape.template to_static<kNDims>());

  kernel.Run(ctx, out, in, {this->window_size_, this->reset_interval_});

  auto ref_tv = TensorView<StorageCPU, float>(this->ref_output_.data(), this->out_shape_);
  Check(out, ref_tv, EqualRelative(1e-5));
}


}  // namespace test
}  // namespace kernels
}  // namespace dali
