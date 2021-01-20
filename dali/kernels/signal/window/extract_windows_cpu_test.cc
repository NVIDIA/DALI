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
#include "dali/kernels/signal/window/extract_windows_cpu.h"
#include "dali/kernels/signal/window/window_functions.h"
#include "dali/kernels/common/utils.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace kernels {
namespace signal {
namespace window {
namespace test {

class ExtractWindowsCpuTest : public::testing::TestWithParam<
  std::tuple<std::array<int64_t, 2>, int64_t, int64_t, int64_t, int64_t, Padding>> {
 public:
  ExtractWindowsCpuTest()
    : data_shape_(std::get<0>(GetParam()))
    , window_length_(std::get<1>(GetParam()))
    , window_step_(std::get<2>(GetParam()))
    , axis_(std::get<3>(GetParam()))
    , window_center_(std::get<4>(GetParam()))
    , padding_(std::get<5>(GetParam()))
    , data_(volume(data_shape_))
    , in_view_(data_.data(), data_shape_) {}

  ~ExtractWindowsCpuTest() override = default;

  template <typename OutputType, typename InputType, int Dims, bool vertical>
  void RunTest();

 protected:
  void SetUp() final {
    SequentialFill(in_view_, 0);
  }
  TensorShape<2> data_shape_;
  int window_length_ = -1, window_step_ = -1, axis_ = -1, window_center_ = -1;
  Padding padding_ = Padding::Zero;
  std::vector<float> data_;
  OutTensorCPU<float, 2> in_view_;
};

template <typename T>
void print_data(const OutTensorCPU<T, 2>& data_view) {
  auto sh = data_view.shape;
  for (int i0 = 0; i0 < sh[0]; i0++) {
    for (int i1 = 0; i1 < sh[1]; i1++) {
      int k = i0 * sh[1] + i1;
      LOG_LINE << " " << data_view.data[k];
    }
    LOG_LINE << "\n";
  }
}

template <typename T>
void print_data(const OutTensorCPU<T, 3>& data_view) {
  auto sh = data_view.shape;
  for (int i0 = 0; i0 < sh[0]; i0++) {
    for (int i1 = 0; i1 < sh[1]; i1++) {
      for (int i2 = 0; i2 < sh[2]; i2++) {
        int k = i0 * sh[1] * sh[2] + i1 * sh[2] + i2;
        LOG_LINE << " " << data_view.data[k];
      }
      LOG_LINE << "\n";
    }
    LOG_LINE << "\n";
  }
  LOG_LINE << "\n";
}

template <typename OutputType, typename InputType, int Dims, bool vertical>
void ExtractWindowsCpuTest::RunTest() {
  constexpr int InputDims = Dims;
  constexpr int OutputDims = Dims + 1;
  ExtractWindowsCpu<OutputType, InputType, Dims, vertical> kernel;
  check_kernel<decltype(kernel)>();

  KernelContext ctx;
  ExtractWindowsArgs args;
  args.window_length = window_length_;
  args.window_step = window_step_;
  args.axis = axis_;
  args.window_center = window_center_;
  args.padding = padding_;

  // Hamming window
  std::vector<float> window_fn_data(window_length_);
  HammingWindow(make_span(window_fn_data));
  auto window_fn_view = OutTensorCPU<float, 1>(window_fn_data.data(), {1});

  KernelRequirements reqs = kernel.Setup(ctx, in_view_, window_fn_view, args);
  auto out_shape = reqs.output_shapes[0][0];

  auto n = in_view_.shape[axis_];
  auto nwindows = padding_ == Padding::None
    ? (n - window_length_) / window_step_ + 1
    : n / window_step_ + 1;
  auto expected_out_shape = vertical ?
    TensorShape<DynamicDimensions>{in_view_.shape[0], window_length_, nwindows} :
    TensorShape<DynamicDimensions>{in_view_.shape[0], nwindows, window_length_};
  ASSERT_EQ(expected_out_shape, out_shape);
  auto expected_out_size = volume(expected_out_shape);
  auto out_size = volume(out_shape);
  ASSERT_EQ(expected_out_size, out_size);

  std::vector<OutputType> expected_out(out_size);
  auto expected_out_view = OutTensorCPU<OutputType, OutputDims>(
    expected_out.data(), out_shape.to_static<OutputDims>());

  TensorShape<> in_shape = in_view_.shape;
  TensorShape<> in_strides = GetStrides(in_shape);

  TensorShape<> flat_out_shape = in_shape;
  flat_out_shape[InputDims-1] = nwindows * window_length_;
  TensorShape<> out_strides = GetStrides(flat_out_shape);

  auto in_stride = in_strides[axis_];
  auto out_stride = out_strides[axis_];

  int window_center_offset = 0;
  if (padding_ != Padding::None)
    window_center_offset = window_center_ < 0 ? window_length_ / 2 : window_center_;
  for (int i = 0; i < in_view_.shape[0]; i++) {
    auto *out_slice = expected_out_view.data + i * out_strides[0];
    auto *in_slice = in_view_.data + i * in_strides[0];
    for (int w = 0; w < nwindows; w++) {
      for (int t = 0; t < window_length_; t++) {
        auto out_k = vertical ? w + t * nwindows : w * window_length_ + t;
        auto in_k = w * window_step_ + t - window_center_offset;
        if (padding_ == Padding::Reflect) {
          while (in_k < 0 || in_k >= n) {
              in_k = (in_k < 0) ? -in_k : 2*n-2-in_k;
          }
        }
        out_slice[out_k] = (in_k >= 0 && in_k < n) ?
          window_fn_data[t] * in_slice[in_k] : 0;
      }
    }
  }


  LOG_LINE << "in:\n";
  print_data(in_view_);

  LOG_LINE << "expected out:\n";
  print_data(expected_out_view);

  std::vector<OutputType> out(out_size);
  auto out_view = OutTensorCPU<OutputType, OutputDims>(
    out.data(), out_shape.to_static<OutputDims>());
  kernel.Run(ctx, out_view, in_view_, window_fn_view, args);

  LOG_LINE << "out:\n";
  print_data(out_view);

  for (int idx = 0; idx < volume(out_view.shape); idx++) {
    ASSERT_EQ(expected_out[idx], out_view.data[idx]) <<
      "Output data doesn't match reference (idx=" << idx << ")";
  }
}

TEST_P(ExtractWindowsCpuTest, Vertical) {
  RunTest<float, float, 2, true>();
}

TEST_P(ExtractWindowsCpuTest, Horizontal) {
  RunTest<float, float, 2, false>();
}

INSTANTIATE_TEST_SUITE_P(ExtractWindowsCpuTest, ExtractWindowsCpuTest, testing::Combine(
    testing::Values(std::array<int64_t, 2>{1, 12},
                    std::array<int64_t, 2>{2, 12}),
    testing::Values(4),  // window_length
    testing::Values(2),  // step
    testing::Values(1),  // axis
    testing::Values(0, 2, 4),  // window offsets
    testing::Values(Padding::None, Padding::Zero, Padding::Reflect)));  // reflect padding

}  // namespace test
}  // namespace window
}  // namespace signal
}  // namespace kernels
}  // namespace dali
