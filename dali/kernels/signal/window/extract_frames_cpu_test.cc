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
#include <complex>
#include <cmath>
#include "dali/kernels/scratch.h"
#include "dali/kernels/signal/signal_kernel_utils.h"
#include "dali/kernels/signal/window/extract_frames_cpu.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace kernels {
namespace signal {
namespace window {
namespace test {

class ExtractFramesCpuTest : public::testing::TestWithParam<
  std::tuple<std::array<int64_t, 2>, int64_t, int64_t, int64_t, int64_t>> {
 public:
  ExtractFramesCpuTest()
    : data_shape_(std::get<0>(GetParam()))
    , window_length_(std::get<1>(GetParam()))
    , window_step_(std::get<2>(GetParam()))
    , in_time_axis_(std::get<3>(GetParam()))
    , out_frame_axis_(std::get<4>(GetParam()))
    , data_(volume(data_shape_))
    , in_view_(data_.data(), data_shape_) {}

  ~ExtractFramesCpuTest() override = default;

 protected:
  void SetUp() final {
    std::mt19937_64 rng;
    UniformRandomFill(in_view_, rng, 0., 1.);
  }
  TensorShape<2> data_shape_;
  int64_t window_length_ = -1, window_step_ = -1, in_time_axis_ = -1, out_frame_axis_ = -1;
  std::vector<float> data_;
  OutTensorCPU<float, 2> in_view_;
};

TEST_P(ExtractFramesCpuTest, ExtractFramesTest) {
  using OutputType = float;
  using InputType = float;
  constexpr int Dims = 2;
  constexpr int InputDims = Dims;
  constexpr int OutputDims = Dims + 1;

  ExtractFramesCpu<OutputType, InputType, Dims> kernel;
  check_kernel<decltype(kernel)>();

  KernelContext ctx;
  ExtractFramesArgs args;
  args.window_length = window_length_;
  args.window_step = window_step_;
  args.in_time_axis = in_time_axis_;
  args.out_frame_axis = out_frame_axis_;

  KernelRequirements reqs = kernel.Setup(ctx, in_view_, args);
  auto shape = reqs.output_shapes[0];

  auto n = in_view_.shape[in_time_axis_];
  auto nwindows = (n + window_step_ - 1) / window_step_;
  auto expected_out_size = nwindows * window_length_ * volume(in_view_.shape) / n;
  auto out_shape = reqs.output_shapes[0][0];
  auto out_size = volume(out_shape);
  ASSERT_EQ(expected_out_size, out_size);

  std::vector<OutputType> expected_out(out_size);
  auto expected_out_view = OutTensorCPU<OutputType, OutputDims>(
    expected_out.data(), out_shape.to_static<OutputDims>());

  TensorShape<> in_shape = in_view_.shape;
  TensorShape<> in_strides = GetStrides(in_shape);

  TensorShape<> out_tmp_shape = in_shape;
  out_tmp_shape[InputDims-1] = nwindows * window_length_;
  TensorShape<> out_strides = GetStrides(out_tmp_shape);

  auto in_stride = in_strides[in_time_axis_];
  auto out_stride = out_strides[in_time_axis_];

  for (int i = 0; i < in_view_.shape[0]; i++) {
    auto *out_slice = expected_out_view.data + i * out_strides[0];
    auto *in_slice = in_view_.data + i * in_strides[0];
    for (int w = 0; w < nwindows; w++) {
      for (int t = 0; t < window_length_; t++) {
        auto out_k = w * window_length_ + t;
        auto in_k = w * window_step_ + t;
        out_slice[out_k] = (in_k < n) ? in_slice[in_k] : 0;
      }
    }
  }

  LOG_LINE << "in:\n";
  for (int i0 = 0; i0 < in_view_.shape[0]; i0++) {
    for (int i1 = 0; i1 < in_view_.shape[1]; i1++) {
      int k = i0*in_shape[1] + i1;
      LOG_LINE << " " << in_view_.data[k];
    }
    LOG_LINE << "\n";
  }
  LOG_LINE << "\n";


  LOG_LINE << "out:\n";
  for (int i0 = 0; i0 < expected_out_view.shape[0]; i0++) {
    for (int i1 = 0; i1 < expected_out_view.shape[1]; i1++) {
      for (int i2 = 0; i2 < expected_out_view.shape[2]; i2++) {
        int k = i0*expected_out_view.shape[1]*expected_out_view.shape[2]+ i1*expected_out_view.shape[2] + i2;
        LOG_LINE << " " << expected_out_view.data[k];
      }
      LOG_LINE << "\n";
    }
    LOG_LINE << "\n";
  }
  LOG_LINE << "\n";


  std::vector<OutputType> out(out_size);
  auto out_view = OutTensorCPU<OutputType, OutputDims>(
    out.data(), out_shape.to_static<OutputDims>());
  kernel.Run(ctx, out_view, in_view_, args);

  for (int idx = 0; idx < volume(out_view.shape); idx++) {
    EXPECT_EQ(expected_out[idx], out_view.data[idx]);
  }
}

INSTANTIATE_TEST_SUITE_P(ExtractFramesCpuTest, ExtractFramesCpuTest, testing::Combine(
    testing::Values(std::array<int64_t, 2>{3, 4}),
                    //std::array<int64_t, 2>{1, 10}),
                    //std::array<int64_t, 2>{1, 64},
                    //std::array<int64_t, 2>{1, 100},
                    //std::array<int64_t, 2>{1, 4096}),
    testing::Values(4, 2),
    testing::Values(2, 4),
    testing::Values(1),
    testing::Values(2)));

}  // namespace test
}  // namespace window
}  // namespace signal
}  // namespace kernels
}  // namespace dali
