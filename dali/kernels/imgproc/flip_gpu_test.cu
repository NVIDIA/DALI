// Copyright (c) 2019, 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/kernels/imgproc/flip_gpu.cuh"
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "dali/kernels/imgproc/flip_test.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace kernels {

class FlipGpuTest: public testing::TestWithParam<std::array<Index, sample_ndim>> {
 public:
  FlipGpuTest()
  : tensor_shape_(GetParam())
  , volume_(volume(tensor_shape_))
  , shape_({tensor_shape_, tensor_shape_, tensor_shape_, tensor_shape_,
           tensor_shape_, tensor_shape_, tensor_shape_, tensor_shape_}) {}

  void SetUp() final {
    ttl_in_.reshape(shape_);
    auto tlv = ttl_in_.cpu(nullptr);
    std::mt19937_64 rng;
    UniformRandomFill(tlv, rng, 0., 10.);
  }

 protected:
  std::vector<int> flip_x_{0, 0, 1, 1, 0, 0, 1, 1};
  std::vector<int> flip_y_{0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<int> flip_z_{0, 0, 0, 0, 1, 1, 1, 1};
  TensorShape<sample_ndim> tensor_shape_;
  size_t volume_;
  TensorListShape<sample_ndim> shape_;
  TestTensorList<float, sample_ndim> ttl_in_;
  TestTensorList<float, sample_ndim> ttl_out_;
};

TEST_P(FlipGpuTest, ImplTest) {
  KernelContext ctx;
  ctx.gpu.stream = 0;
  FlipGPU<float> kernel;
  auto in_view = ttl_in_.gpu(nullptr);
  ttl_in_.invalidate_cpu();
  KernelRequirements reqs = kernel.Setup(ctx, in_view);
  ttl_out_.reshape(reqs.output_shapes[0].to_static<sample_ndim>());
  auto out_view = ttl_out_.gpu();
  for (int i = 0; i < in_view.num_samples(); ++i) {
    detail::gpu::FlipImpl(
        out_view.tensor_data(i), in_view.tensor_data(i),
        tensor_shape_, flip_z_[i], flip_y_[i], flip_z_[i], nullptr);
  }
  kernel.Run(ctx, out_view, in_view, flip_z_, flip_y_, flip_x_);
  auto out_view_cpu = ttl_out_.cpu(nullptr);
  auto in_view_cpu = ttl_in_.cpu(nullptr);
  for (int i = 0; i < out_view_cpu.num_samples(); ++i) {
    ASSERT_TRUE(is_flipped(out_view_cpu.tensor_data(i),
                           in_view_cpu.tensor_data(i),
                           shape_[i][0], shape_[i][1], shape_[i][2], shape_[i][3], shape_[i][4],
                           flip_z_[i], flip_y_[i], flip_x_[i]));
  }
}

TEST_P(FlipGpuTest, KernelTest) {
  KernelContext ctx;
  ctx.gpu.stream = 0;
  FlipGPU<float> kernel;
  auto in_view = ttl_in_.gpu(nullptr);
  ttl_in_.invalidate_cpu();
  KernelRequirements reqs = kernel.Setup(ctx, in_view);
  ttl_out_.reshape(reqs.output_shapes[0].to_static<sample_ndim>());
  auto out_view = ttl_out_.gpu();
  kernel.Run(ctx, out_view, in_view, flip_z_, flip_y_, flip_x_);
  auto out_view_cpu = ttl_out_.cpu(nullptr);
  auto in_view_cpu = ttl_in_.cpu(nullptr);
  for (int i = 0; i < out_view_cpu.num_samples(); ++i) {
    ASSERT_TRUE(is_flipped(out_view_cpu.tensor_data(i),
                           in_view_cpu.tensor_data(i),
                           shape_[i][0], shape_[i][1], shape_[i][2], shape_[i][3], shape_[i][4],
                           flip_z_[i], flip_y_[i], flip_x_[i]));
  }
}

INSTANTIATE_TEST_SUITE_P(FlipGpuTest, FlipGpuTest,
    ::testing::ValuesIn({
        std::array<Index, sample_ndim>{4, 1, 2, 2, 10},
        std::array<Index, sample_ndim>{4, 1, 2, 2, 2},
        std::array<Index, sample_ndim>{1, 4, 9, 18, 3},
        std::array<Index, sample_ndim>{1, 3, 18, 9, 4}}));

}  // namespace kernels
}  // namespace dali
