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
#include "dali/core/geom/mat.h"
#include "dali/core/tensor_shape.h"
#include "dali/kernels/common/copy.h"
#include "dali/test/mat2tensor.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/kernels/test/kernel_test_utils.h"
#include "dali/kernels/imgproc/pointwise/linear_transformation_gpu.h"
#include "dali/test/cv_mat_utils.h"
#include "dali/kernels/imgproc/roi.h"
#include "dali/kernels/dynamic_scratchpad.h"

namespace dali {
namespace kernels {
namespace test {


namespace {

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
    CUDA_CALL(cudaDeviceSynchronize());
  }

  void TearDown() final {
    CUDA_CALL(cudaFree(input_device_));
    CUDA_CALL(cudaFree(output_));
  }


  In *input_device_;
  Out *output_;
  std::vector<In> input_host_;
  std::vector<float> ref_output_;
  std::vector<TensorShape<kNDims>> in_shapes_ = {{4, 3, kNChannelsIn}, {4, 3, kNChannelsIn}};
  std::vector<TensorShape<kNDims>> out_shapes_ = {{4, 3, kNChannelsOut}, {4, 3, kNChannelsOut}};
  mat<kNChannelsOut, kNChannelsIn, float> mat_{{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}};
  vec<kNChannelsOut, float> vec_{42, 69};
  std::vector<mat<kNChannelsOut, kNChannelsIn, float>> vmat_ = {mat_, mat_ + 1.f};
  std::vector<vec<kNChannelsOut, float>> vvec_ = {vec_, vec_ + 1.f};
  std::vector<Roi<2>> rois_ = {{{1, 1}, {2, 2}},
                               {{0, 1}, {1, 2}}};


  void calc_output() {
    for (size_t i = 0; i < input_host_.size(); i += kNChannelsIn) {
      for (size_t j = 0; j < kNChannelsOut; j++) {
        float res = vec_.v[j];
        for (size_t k = 0; k < kNChannelsIn; k++) {
          res += static_cast<float>(input_host_[i + k]) * mat_.at(j, k);
        }
        ref_output_.push_back(res);
      }
    }
  }


  size_t dataset_size(const std::vector<TensorShape<kNDims>> &shapes) {
    int ret = 0;
    for (const auto &sh : shapes) {
      ret += volume(sh);
    }
    return ret;
  }
};

using TestTypes = std::tuple<float>;
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
  ctx.gpu.stream = 0;
  InListGPU<typename TypeParam::In, kNDims> in(this->input_device_, this->in_shapes_);
  auto reqs = kernel.Setup(ctx, in, make_cspan(this->vmat_), make_cspan(this->vvec_));
  ASSERT_EQ(this->out_shapes_.size(), static_cast<size_t>(reqs.output_shapes[0].num_samples()))
                        << "Kernel::Setup provides incorrect shape";
  for (size_t i = 0; i < this->out_shapes_.size(); i++) {
    EXPECT_EQ(this->out_shapes_[i], reqs.output_shapes[0][i])
                  << "Kernel::Setup provides incorrect shape";
  }
}


TYPED_TEST(LinearTransformationGpuTest, setup_test_with_roi) {
  TheKernel<TypeParam> kernel;
  KernelContext ctx;
  ctx.gpu.stream = 0;
  InListGPU<typename TypeParam::In, kNDims> in(this->input_device_, this->in_shapes_);
  auto reqs = kernel.Setup(ctx, in, make_cspan(this->vmat_), make_cspan(this->vvec_),
                           make_cspan(this->rois_));
  auto ref_shape = ShapeFromRoi(this->rois_[0], kNChannelsOut);
  ASSERT_EQ(ref_shape, reqs.output_shapes[0][0]);
}


TYPED_TEST(LinearTransformationGpuTest, run_test) {
  TheKernel<TypeParam> kernel;
  KernelContext ctx;
  ctx.gpu.stream = 0;
  InListGPU<typename TypeParam::In, kNDims> in(this->input_device_, this->in_shapes_);

  auto reqs = kernel.Setup(ctx, in, make_cspan(this->vmat_), make_cspan(this->vvec_));

  DynamicScratchpad dyn_scratchpad(AccessOrder(ctx.gpu.stream));
  ctx.scratchpad = &dyn_scratchpad;

  OutListGPU<typename TypeParam::Out, kNDims> out(
          this->output_, reqs.output_shapes[0].template to_static<kNDims>());

  kernel.Run(ctx, out, in, make_cspan(this->vmat_), make_cspan(this->vvec_));
  CUDA_CALL(cudaDeviceSynchronize());

  auto res = copy<mm::memory_kind::host>(out[0]);
  auto ref_tv = TensorView<StorageCPU, typename TypeParam::Out>(this->ref_output_.data(),
                                                                this->out_shapes_[0]);
  Check(res.first, ref_tv, EqualUlp());
}


TYPED_TEST(LinearTransformationGpuTest, run_test_with_roi) {
  TheKernel<TypeParam> kernel;
  KernelContext ctx;
  ctx.gpu.stream = 0;
  InListGPU<typename TypeParam::In, kNDims> in(this->input_device_, this->in_shapes_);

  auto reqs = kernel.Setup(ctx, in,
                           make_cspan(this->vmat_), make_cspan(this->vvec_),
                           make_cspan(this->rois_));

  DynamicScratchpad dyn_scratchpad(AccessOrder(ctx.gpu.stream));
  ctx.scratchpad = &dyn_scratchpad;

  OutListGPU<typename TypeParam::Out, kNDims> out(
          this->output_, reqs.output_shapes[0].template to_static<kNDims>());

  kernel.Run(ctx, out, in,
             make_cspan(this->vmat_), make_cspan(this->vvec_), make_cspan(this->rois_));
  CUDA_CALL(cudaDeviceSynchronize());

  auto res = copy<mm::memory_kind::host>(out[0]);
  auto mat = testing::copy_to_mat<kNChannelsOut>(
      this->rois_[0],
      this->ref_output_.data(),
      this->out_shapes_[0][0],
      this->out_shapes_[0][1]);
  Check(view_as_tensor<typename TypeParam::Out>(mat), res.first, EqualUlp());
}

}  // namespace test
}  // namespace kernels
}  // namespace dali

