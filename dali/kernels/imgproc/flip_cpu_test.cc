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
#include <tuple>
#include <vector>
#include "dali/kernels/imgproc/flip_test.h"
#include "dali/kernels/imgproc/flip_cpu.h"
#include "dali/kernels/test/test_tensors.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/kernels/test/tensor_test_utils.h"

namespace dali {
namespace kernels {

class FlipCpuTest
    : public ::testing::TestWithParam<std::tuple<int, int, int, std::array<Index, 4>>> {
 public:
  FlipCpuTest()
  : flip_x(std::get<0>(GetParam()))
  , flip_y(std::get<1>(GetParam()))
  , flip_z(std::get<2>(GetParam()))
  , shape(std::get<3>(GetParam()))
  , data(volume(shape))
  , in_view(data.data(), shape) {}

  ~FlipCpuTest() override = default;

 protected:
  void SetUp() override {
    std::mt19937_64 rng;
    UniformRandomFill(in_view, rng, 0., 10.);
  }

  int flip_x;
  int flip_y;
  int flip_z;
  kernels::TensorShape<4> shape;
  std::vector<float> data;
  OutTensorCPU<float, 4> in_view;
};

TEST_P(FlipCpuTest, BasicTest) {
  KernelContext ctx;
  FlipCPU<float> kernel;
  KernelRequirements reqs = kernel.Setup(ctx, in_view);
  auto out_shape = reqs.output_shapes[0][0].to_static<4>();
  std::vector<float> out_data(volume(out_shape));
  auto out_view = OutTensorCPU<float, 4>(out_data.data(), out_shape);
  kernel.Run(ctx, out_view, in_view, flip_z, flip_y, flip_x);
  ASSERT_TRUE(is_flipped(out_view.data, in_view.data,
        shape[0], shape[1], shape[2], shape[3], flip_z, flip_y, flip_x));
}

INSTANTIATE_TEST_SUITE_P(FlipCpuTest, FlipCpuTest, testing::Combine(
    testing::Values(0, 1),
    testing::Values(0, 1),
    testing::Values(0, 1),
    testing::Values(std::array<Index, 4>{8, 9, 9, 3}, std::array<Index, 4>{3, 18, 18, 2})));

}  // namespace kernels
}  // namespace dali
