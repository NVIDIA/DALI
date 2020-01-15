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
//#include <vector>
//#include <tuple>
//#include "dali/core/geom/mat.h"
//#include "dali/core/tensor_shape.h"
//#include "dali/kernels/scratch.h"
//#include "dali/kernels/common/copy.h"
//#include "dali/test/mat2tensor.h"
#include "dali/test/tensor_test_utils.h"
//#include "dali/kernels/test/kernel_test_utils.h"
//#include "dali/kernels/imgproc/pointwise/linear_transformation_cpu.h"
//#include "dali/test/cv_mat_utils.h"
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
    input_.resize(dataset_size(shape_));
  }


  void SetUp() final {
    std::mt19937_64 rng;
    UniformRandomFill(input_, rng, -100., 100.);
    calc_output();
  }


  std::vector<InputType> input_;
  std::vector<float> ref_output_;
  int window_size_ = 2048;
  int buffer_length_ = 32000;
  TensorShape<kNDims> shape_ = {buffer_length_};


  void calc_output() {
    ref_output_.resize(dataset_size(shape_));
    for (int i = 0; i < buffer_length_; i++) {
      float sumsq = 0;
      for (int j = 0; j < window_size_ && i + j < buffer_length_; j++) {
        sumsq += input_[i + j] * input_[i + j];
      }
      ref_output_[i] = sumsq / window_size_;
    }
  }


  size_t dataset_size(const TensorShape<kNDims> &shape) {
    return volume(shape);
  }
};

using TestTypes = ::testing::Types<int/*uint8_t, int8_t, uint16_t, int16_t, int32_t, float*/>;

TYPED_TEST_SUITE(MovingMeanSquareCpuTest, TestTypes);

namespace {

template<class GtestTypeParam>
using TestedKernel = signal::MovingMeanSquareCpu<GtestTypeParam>;

}  // namespace


TYPED_TEST(MovingMeanSquareCpuTest, CheckKernel) {
  check_kernel<TestedKernel<TypeParam>>();
  SUCCEED();
}


TYPED_TEST(MovingMeanSquareCpuTest, SetupTest) {
  TestedKernel<TypeParam> kernel;
  KernelContext ctx;
  InTensorCPU<TypeParam, kNDims> in(this->input_.data(), this->shape_);
  auto reqs = kernel.Setup(ctx, in, {this->window_size_});
  ASSERT_EQ(this->shape_, reqs.output_shapes[0][0]) << "Kernel::Setup provides incorrect shape";
}


TYPED_TEST(MovingMeanSquareCpuTest, RunTest) {
  TestedKernel<TypeParam> kernel;
  KernelContext ctx;
  InTensorCPU<TypeParam, kNDims> in(this->input_.data(), this->shape_);

  auto reqs = kernel.Setup(ctx, in, {this->window_size_});

  auto out_shape = reqs.output_shapes[0][0];
  std::vector<float> output;
  output.resize(dali::volume(out_shape));
  OutTensorCPU<float, kNDims> out(output.data(), out_shape.template to_static<kNDims>());

  kernel.Run(ctx, out, in, {this->window_size_});

  auto ref_tv = TensorView<StorageCPU, float>(this->ref_output_.data(), this->shape_);
  Check(out, ref_tv, EqualUlp());
}


}  // namespace test
}  // namespace kernels
}  // namespace dali

