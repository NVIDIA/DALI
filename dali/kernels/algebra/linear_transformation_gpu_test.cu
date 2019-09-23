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
#include "dali/kernels/algebra/linear_transformation_gpu.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/tensor_shape.h"
#include "dali/kernels/common/copy.h"
#include "dali/kernels/test/tensor_test_utils.h"
#include "dali/kernels/test/kernel_test_utils.h"
#include "dali/kernels/imgproc/color_manipulation/color_manipulation_test_utils.h"

namespace dali {
namespace kernels {
namespace linear_transformation {
namespace test {


namespace {

/**
 * Rounding to nearest even (like GPU does it)
 */
template <class Out>
std::enable_if_t<std::is_integral<Out>::value, Out> custom_round(float val) {
  return ConvertSat<Out>(std::nearbyint(val));
}


template <class Out>
std::enable_if_t<!std::is_integral<Out>::value, Out> custom_round(float val) {
  return val;
}


constexpr int kNDims = 3;
constexpr int kNChannelsIn = 5;
constexpr int kNChannelsOut = 2;


}  // namespace

template <class InputOutputTypes>
class LinearTransformationGpuTest : public ::testing::Test {
  using In = typename InputOutputTypes::In;
  using Out = typename InputOutputTypes::Out;

 public:
  LinearTransformationGpuTest() {
    input_host_.resize(dataset_size(in_shapes_));
  }


  void SetUp() final {
    std::mt19937_64 rng;
    UniformRandomFill(input_host_, rng, 0., 10.);
    calc_output();
    CUDA_CALL(cudaMalloc(&input_device_, sizeof(In) * dataset_size(in_shapes_)));
    CUDA_CALL(cudaMemcpy(input_device_, input_host_.data(), input_host_.size() * sizeof(In),
                         cudaMemcpyDefault));
    CUDA_CALL(cudaMalloc(&output_, dataset_size(out_shapes_) * sizeof(Out)));
    cudaDeviceSynchronize();
  }


  In *input_device_;
  Out *output_;
  std::vector<In> input_host_;
  std::vector<Out> ref_output_;
  std::vector<TensorShape<kNDims>> in_shapes_ = {{4, 3, kNChannelsIn}};
  std::vector<TensorShape<kNDims>> out_shapes_ = {{4, 3, kNChannelsOut}};
  mat<kNChannelsOut, kNChannelsIn, float> mat_{{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}};
  vec<kNChannelsOut, float> vec_{42, 69};
  std::vector<mat<kNChannelsOut, kNChannelsIn, float>> vmat_ = {mat_};
  std::vector<vec<kNChannelsOut, float>> vvec_ = {vec_};
  std::vector<Roi<2>> rois_ = {{{1, 1}, {2, 2}}};


  void calc_output() {
    for (size_t i = 0; i < input_host_.size(); i += kNChannelsIn) {
      for (size_t j = 0; j < kNChannelsOut; j++) {
        float res = vec_.v[j];
        for (size_t k = 0; k < kNChannelsIn; k++) {
          res += static_cast<float>(input_host_[i + k]) * mat_.at(j, k);
        }
        ref_output_.push_back(custom_round<Out>(res));
      }
    }
  }


  size_t dataset_size(const std::vector<TensorShape<kNDims>> &shapes) {
    int ret = 0;
    for (auto sh : shapes) {
      ret += volume(sh);
    }
    return ret;
  }
};

using TestTypes = std::tuple<uint8_t>;
/* Cause the line below takes RIDICULOUSLY long time to compile */
// using TestTypes = std::tuple<uint8_t, int8_t, uint16_t, int16_t, int32_t, float>;

INPUT_OUTPUT_TYPED_TEST_SUITE(LinearTransformationGpuTest, TestTypes);

namespace {

template <class GtestTypeParam>
using TheKernel = LinearTransformationGpu<typename GtestTypeParam::Out, typename GtestTypeParam::In,
        kNChannelsOut, kNChannelsIn, kNDims - 1>;

}  // namespace


TYPED_TEST(LinearTransformationGpuTest, check_kernel) {
  check_kernel<TheKernel<TypeParam>>();
}


TYPED_TEST(LinearTransformationGpuTest, setup_test) {
  TheKernel<TypeParam> kernel;
  KernelContext ctx;
  InListGPU<typename TypeParam::In, kNDims> in(this->input_device_, this->in_shapes_);
  auto reqs = kernel.Setup(ctx, in, this->vmat_, this->vvec_);
  ASSERT_EQ(this->out_shapes_.size(), static_cast<size_t>(reqs.output_shapes[0].num_samples()))
                        << "Kernel::Setup provides incorrect shape";
  for (size_t i = 0; i < this->out_shapes_.size(); i++) {
    EXPECT_EQ(this->out_shapes_[i], reqs.output_shapes[0][i])
                  << "Kernel::Setup provides incorrect shape";
  }
}


TYPED_TEST(LinearTransformationGpuTest, run_test) {
  TheKernel<TypeParam> kernel;
  KernelContext c;
  InListGPU<typename TypeParam::In, kNDims> in(this->input_device_, this->in_shapes_);

  auto reqs = kernel.Setup(c, in, this->vmat_, this->vvec_);

  ScratchpadAllocator sa;
  sa.Reserve(reqs.scratch_sizes);
  auto scratchpad = sa.GetScratchpad();
  c.scratchpad = &scratchpad;

  OutListGPU<typename TypeParam::Out, kNDims> out(
          this->output_, reqs.output_shapes[0].template to_static<kNDims>());

  kernel.Run(c, out, in, this->vmat_, this->vvec_);
  cudaDeviceSynchronize();

  auto res = copy<AllocType::Host>(out[0]);
  ASSERT_EQ(static_cast<int>(this->ref_output_.size()), res.first.num_elements());
  for (size_t i = 0; i < this->ref_output_.size(); i++) {
    EXPECT_FLOAT_EQ(this->ref_output_[i], res.second.get()[i]) << "Failed for index " << i;
  }
}


TYPED_TEST(LinearTransformationGpuTest, run_test_with_roi) {
  TheKernel<TypeParam> kernel;
  KernelContext c;
  InListGPU<typename TypeParam::In, kNDims> in(this->input_device_, this->in_shapes_);

  auto reqs = kernel.Setup(c, in, this->vmat_, this->vvec_, this->rois_);

  ScratchpadAllocator sa;
  sa.Reserve(reqs.scratch_sizes);
  auto scratchpad = sa.GetScratchpad();
  c.scratchpad = &scratchpad;

  OutListGPU<typename TypeParam::Out, kNDims> out(
          this->output_, reqs.output_shapes[0].template to_static<kNDims>());

  kernel.Run(c, out, in, this->vmat_, this->vvec_, this->rois_);
  cudaDeviceSynchronize();

  auto res = copy<AllocType::Host>(out[0]);
  auto mat = color_manipulation::test::to_mat<kNChannelsOut>(this->ref_output_.data(),
                                                             this->rois_[0],
                                                             this->out_shapes_[0][0],
                                                             this->out_shapes_[0][1]);

  ASSERT_EQ(mat.rows * mat.cols * mat.channels(), res.first.num_elements())
                        << "Number of elements doesn't match";
  auto ptr = reinterpret_cast<typename TypeParam::Out *>(mat.data);
  for (int i = 0; i < res.first.num_elements(); i++) {
    EXPECT_FLOAT_EQ(ptr[i], res.second.get()[i]) << "Failed at idx: " << i;
  }
}


namespace {

template <int ndims>
bool cmp_shapes(const TensorShape<ndims> &lhs, ivec<ndims - 1> rhs) {
  std::reverse(rhs.begin(), rhs.end());
  for (size_t i = 0; i < rhs.size(); i++) {
    if (lhs[i] != rhs[i]) return false;
  }
  return true;
}

}  // namespace


TYPED_TEST(LinearTransformationGpuTest, sample_descriptors) {
  using In = typename TypeParam::In;
  using Out = typename TypeParam::Out;

  InListGPU<In, kNDims> in(this->input_device_, this->in_shapes_);
  OutListGPU<Out, kNDims> out(this->output_, TensorListShape<3>(this->out_shapes_));

  auto res = detail::CreateSampleDescriptors(out, in, this->vmat_, this->vvec_, this->rois_);

  EXPECT_EQ(this->input_device_, res[0].in);
  EXPECT_EQ(this->output_, res[0].out);
  EXPECT_TRUE(cmp_shapes<kNDims>(this->in_shapes_[0], res[0].in_size));
  EXPECT_TRUE(cmp_shapes<kNDims>(this->out_shapes_[0], res[0].out_size));
  EXPECT_EQ(this->vmat_[0], res[0].transformation_matrix);
}


}  // namespace test
}  // namespace linear_transformation
}  // namespace kernels
}  // namespace dali

