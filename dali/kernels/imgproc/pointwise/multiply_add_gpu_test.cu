// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/tensor_shape.h"
#include "dali/kernels/common/copy.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/kernels/test/kernel_test_utils.h"
#include "dali/kernels/imgproc/pointwise/multiply_add_gpu.h"
#include "dali/kernels/dynamic_scratchpad.h"

namespace dali {
namespace kernels {
namespace multiply_add {
namespace test {

namespace {

constexpr size_t kNdims = 3;


/**
 * Rounding to nearest even (like GPU does it)
 */
template <class In, class Out>
std::enable_if_t<std::is_integral<Out>::value, Out> custom_round(float val) {
  return static_cast<Out>(std::nearbyint(val));
}


template <class In, class Out>
std::enable_if_t<!std::is_integral<Out>::value, Out> custom_round(float val) {
  return val;
}


}  // namespace

template <class InputOutputTypes>
class MultiplyAddGpuTest : public ::testing::Test {
  using In = typename InputOutputTypes::In;
  using Out = typename InputOutputTypes::Out;

 public:
  MultiplyAddGpuTest() {
    input_host_.resize(dataset_size());
  }


  void SetUp() final {
    std::mt19937_64 rng;
    UniformRandomFill(input_host_, rng, 0., 10.);
    calc_output(0);
    CUDA_CALL(cudaMalloc(&input_device_, sizeof(In) * dataset_size()));
    CUDA_CALL(cudaMemcpy(input_device_, input_host_.data(), input_host_.size() * sizeof(In),
                         cudaMemcpyDefault));
    CUDA_CALL(cudaMalloc(&output_, dataset_size() * sizeof(Out)));
    CUDA_CALL(cudaDeviceSynchronize());

    verify_test();
  }


  In *input_device_;
  Out *output_;
  std::vector<In> input_host_;
  std::vector<Out> ref_output_;
  std::vector<TensorShape<kNdims>> shapes_ = {{480, 640, 3}};
  std::vector<float> addends_ = {4};
  std::vector<float> multipliers_ = {3};


  void verify_test() {
    assert(shapes_.size() == addends_.size());
    assert(addends_.size() == multipliers_.size());
    assert(dataset_size() == input_host_.size());
    assert(dataset_size() == ref_output_.size());
  }


  void calc_output(int idx) {
    for (auto in : input_host_) {
      ref_output_.push_back(custom_round<In, Out>(in * multipliers_[idx] + addends_[idx]));
    }
  }


  size_t dataset_size() {
    int ret = 0;
    for (const auto &sh : shapes_) {
      ret += volume(sh);
    }
    return ret;
  }
};

using TestTypes = std::tuple<int8_t, float>;
/* Cause the line below takes RIDICULOUSLY long time to compile */
// using TestTypes = std::tuple<uint8_t, int8_t, uint16_t, int16_t, int32_t, float>;

INPUT_OUTPUT_TYPED_TEST_SUITE(MultiplyAddGpuTest, TestTypes);

namespace {

template <class GtestTypeParam>
using TheKernel = MultiplyAddGpu
        <typename GtestTypeParam::Out, typename GtestTypeParam::In, kNdims>;

}  // namespace


TYPED_TEST(MultiplyAddGpuTest, check_kernel) {
  check_kernel<TheKernel<TypeParam>>();
}


TYPED_TEST(MultiplyAddGpuTest, setup_test) {
  TheKernel<TypeParam> kernel;
  KernelContext ctx;
  ctx.gpu.stream = 0;
  InListGPU<typename TypeParam::In, kNdims> in(this->input_device_, this->shapes_);
  auto reqs = kernel.Setup(ctx, in, this->addends_, this->multipliers_);
  ASSERT_EQ(this->shapes_.size(), static_cast<size_t>(reqs.output_shapes[0].num_samples()))
                        << "Kernel::Setup provides incorrect shape";
  for (size_t i = 0; i < this->shapes_.size(); i++) {
    EXPECT_EQ(this->shapes_[i], reqs.output_shapes[0][i])
                  << "Kernel::Setup provides incorrect shape";
  }
}


TYPED_TEST(MultiplyAddGpuTest, run_test) {
  TheKernel<TypeParam> kernel;
  KernelContext ctx;
  ctx.gpu.stream = 0;
  InListGPU<typename TypeParam::In, kNdims> in(this->input_device_, this->shapes_);
  OutListGPU<typename TypeParam::Out, kNdims> out(this->output_,
                                                  TensorListShape<kNdims>(this->shapes_));

  auto reqs = kernel.Setup(ctx, in, this->addends_, this->multipliers_);

  DynamicScratchpad dyn_scratchpad(AccessOrder(ctx.gpu.stream));
  ctx.scratchpad = &dyn_scratchpad;

  kernel.Run(ctx, out, in, this->addends_, this->multipliers_);
  CUDA_CALL(cudaDeviceSynchronize());

  auto res = copy<mm::memory_kind::host>(out[0]);
  ASSERT_EQ(static_cast<int>(this->ref_output_.size()), res.first.num_elements());
  for (size_t i = 0; i < this->ref_output_.size(); i++) {
    EXPECT_FLOAT_EQ(this->ref_output_[i], res.second.get()[i]) << "Failed for index " << i;
  }
}


TYPED_TEST(MultiplyAddGpuTest, sample_descriptors) {
  using InType = typename TypeParam::In;
  using OutType = typename TypeParam::Out;
  InListGPU<InType, kNdims> in(this->input_device_, this->shapes_);
  OutListGPU<OutType, kNdims> out(this->output_, TensorListShape<3>(this->shapes_));
  std::vector<SampleDescriptor<OutType, InType, kNdims-1>> res(in.num_samples());
  CreateSampleDescriptors(make_span(res), out, in, this->addends_, this->multipliers_);
  EXPECT_EQ(this->input_device_, res[0].in);
  EXPECT_EQ(this->output_, res[0].out);
  ivec<kNdims - 2> ref_pitch = {1920};
  EXPECT_EQ(ref_pitch, res[0].in_pitch);
  EXPECT_EQ(ref_pitch, res[0].out_pitch);
  EXPECT_EQ(this->addends_[0], res[0].addend);
  EXPECT_EQ(this->multipliers_[0], res[0].multiplier);
}


}  // namespace test
}  // namespace multiply_add
}  // namespace kernels
}  // namespace dali
