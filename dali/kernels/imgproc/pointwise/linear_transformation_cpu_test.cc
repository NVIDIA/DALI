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
#include "dali/core/geom/mat.h"
#include "dali/core/tensor_shape.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/common/copy.h"
#include "dali/test/mat2tensor.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/kernels/test/kernel_test_utils.h"
#include "dali/kernels/imgproc/pointwise/linear_transformation_cpu.h"
#include "dali/test/cv_mat_utils.h"

namespace dali {
namespace kernels {
namespace test {


namespace {

constexpr int kNDims = 3;
constexpr int kNChannelsIn = 5;
constexpr int kNChannelsOut = 2;

}  // namespace

template <class InputOutputTypes>
class LinearTransformationCpuTest : public ::testing::Test {
  using In = typename InputOutputTypes::In;
  using Out = typename InputOutputTypes::Out;

 public:
  LinearTransformationCpuTest() {
    input_.resize(dataset_size(in_shape_));
  }


  void SetUp() final {
    std::mt19937_64 rng;
    UniformRandomFill(input_, rng, 0., 10.);
    calc_output();
  }


  std::vector<In> input_;
  std::vector<float> ref_output_;
  TensorShape<kNDims> in_shape_ = {9, 12, kNChannelsIn};
  TensorShape<kNDims> out_shape_ = {9, 12, kNChannelsOut};
  mat<kNChannelsOut, kNChannelsIn, float> mat_{{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}};
  vec<kNChannelsOut, float> vec_{42, 69};
  Roi<2> roi_ = {{1, 1},
                 {3, 5}};


  void calc_output() {
    for (size_t i = 0; i < input_.size(); i += kNChannelsIn) {
      for (size_t j = 0; j < kNChannelsOut; j++) {
        float res = vec_.v[j];
        for (size_t k = 0; k < kNChannelsIn; k++) {
          res += static_cast<float>(input_[i + k]) * mat_.at(j, k);
        }
        ref_output_.push_back(res);
      }
    }
  }


  size_t dataset_size(const TensorShape<kNDims> &shape) {
    return volume(shape);
  }
};

using TestTypes = std::tuple<uint8_t, int8_t, uint16_t, int16_t, int32_t, float>;

INPUT_OUTPUT_TYPED_TEST_SUITE(LinearTransformationCpuTest, TestTypes);

namespace {

template <class GtestTypeParam>
using TheKernel = LinearTransformationCpu<typename GtestTypeParam::Out, typename GtestTypeParam::In,
        kNChannelsOut, kNChannelsIn, kNDims>;

}  // namespace


TYPED_TEST(LinearTransformationCpuTest, check_kernel) {
  check_kernel<TheKernel<TypeParam>>();
}


TYPED_TEST(LinearTransformationCpuTest, setup_test) {
  TheKernel<TypeParam> kernel;
  KernelContext ctx;
  InTensorCPU<typename TypeParam::In, kNDims> in(this->input_.data(), this->in_shape_);
  auto reqs = kernel.Setup(ctx, in, this->mat_, this->vec_);
  ASSERT_EQ(this->out_shape_, reqs.output_shapes[0][0]) << "Kernel::Setup provides incorrect shape";
}

TYPED_TEST(LinearTransformationCpuTest, setup_test_with_roi) {
  TheKernel<TypeParam> kernel;
  KernelContext ctx;
  InTensorCPU<typename TypeParam::In, kNDims> in(this->input_.data(), this->in_shape_);
  auto reqs = kernel.Setup(ctx, in, this->mat_, this->vec_, &this->roi_);
  auto ref_shape = ShapeFromRoi(this->roi_, kNChannelsOut);
  ASSERT_EQ(ref_shape, reqs.output_shapes[0][0]);
}


TYPED_TEST(LinearTransformationCpuTest, run_test) {
  TheKernel<TypeParam> kernel;
  KernelContext ctx;
  InTensorCPU<typename TypeParam::In, kNDims> in(this->input_.data(), this->in_shape_);

  auto reqs = kernel.Setup(ctx, in, this->mat_, this->vec_);

  auto out_shape = reqs.output_shapes[0][0];
  std::vector<typename TypeParam::Out> output;
  output.resize(dali::volume(out_shape));
  OutTensorCPU<typename TypeParam::Out, kNDims> out(output.data(),
                                                    out_shape.template to_static<kNDims>());

  kernel.Run(ctx, out, in, this->mat_, this->vec_);

  auto ref_tv = TensorView<StorageCPU, float>(this->ref_output_.data(), this->out_shape_);
  Check(out, ref_tv, EqualUlp());
}


TYPED_TEST(LinearTransformationCpuTest, run_test_with_roi) {
  TheKernel<TypeParam> kernel;
  KernelContext ctx;
  InTensorCPU<typename TypeParam::In, kNDims> in(this->input_.data(), this->in_shape_);

  auto reqs = kernel.Setup(ctx, in, this->mat_, this->vec_, &this->roi_);

  auto out_shape = reqs.output_shapes[0][0];
  std::vector<typename TypeParam::Out> output;
  output.resize(dali::volume(out_shape));
  OutTensorCPU<typename TypeParam::Out, kNDims> out(output.data(),
                                                    out_shape.template to_static<kNDims>());

  kernel.Run(ctx, out, in, this->mat_, this->vec_, &this->roi_);


  auto mat = testing::copy_to_mat<kNChannelsOut>(
      this->roi_,
      this->ref_output_.data(),
      this->in_shape_[0],
      this->in_shape_[1]);
  Check(out, view_as_tensor<float>(mat), EqualUlp());
}

}  // namespace test
}  // namespace kernels
}  // namespace dali

