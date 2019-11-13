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
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace kernels {

class FlipCpuTest
    : public::testing::TestWithParam<std::tuple<int, int, int, std::array<Index, sample_ndim>>> {
 public:
  FlipCpuTest()
  : flip_x_(std::get<0>(GetParam()))
  , flip_y_(std::get<1>(GetParam()))
  , flip_z_(std::get<2>(GetParam()))
  , shape_(std::get<3>(GetParam()))
  , data_(volume(shape_))
  , in_view_(data_.data(), shape_) {}

  ~FlipCpuTest() override = default;

 protected:
  void SetUp() final {
    std::mt19937_64 rng;
    UniformRandomFill(in_view_, rng, 0., 10.);
  }

  int flip_x_;
  int flip_y_;
  int flip_z_;
  TensorShape<sample_ndim> shape_;
  std::vector<float> data_;
  OutTensorCPU<float, sample_ndim> in_view_;
};

TEST_P(FlipCpuTest, ImplTest) {
  std::vector<float> out_data(volume(shape_));
  detail::cpu::FlipImpl(
      out_data.data(), in_view_.data,
      shape_, flip_z_, flip_y_, flip_x_);
  ASSERT_TRUE(is_flipped(out_data.data(), in_view_.data,
                         shape_[0], shape_[1], shape_[2], shape_[3], shape_[4],
                         flip_z_, flip_y_, flip_x_));
}

TEST_P(FlipCpuTest, KernelTest) {
  KernelContext ctx;
  FlipCPU<float> kernel;
  KernelRequirements reqs = kernel.Setup(ctx, in_view_);
  auto out_shape = reqs.output_shapes[0][0].to_static<sample_ndim>();
  std::vector<float> out_data(volume(out_shape));
  auto out_view = OutTensorCPU<float, sample_ndim>(out_data.data(), out_shape);
  kernel.Run(ctx, out_view, in_view_, flip_z_, flip_y_, flip_x_);
  ASSERT_TRUE(is_flipped(out_view.data, in_view_.data,
                         shape_[0], shape_[1], shape_[2], shape_[3], shape_[4],
                         flip_z_, flip_y_, flip_x_));
}

INSTANTIATE_TEST_SUITE_P(FlipCpuTest, FlipCpuTest, testing::Combine(
    testing::Values(0, 1),
    testing::Values(0, 1),
    testing::Values(0, 1),
    testing::Values(std::array<Index, sample_ndim>{1, 8, 9, 9, 3},
                    std::array<Index, sample_ndim>{2, 3, 18, 18, 2})));

}  // namespace kernels
}  // namespace dali
