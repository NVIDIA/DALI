// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include <cmath>
#include <complex>
#include <tuple>
#include <vector>

#include "dali/core/boundary.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/imgproc/convolution/convolution_cpu.h"
#include "dali/kernels/scratch.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace kernels {

template <typename T>
struct CyclicPixelWrapperTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(CyclicPixelWrapperTest);

template <int num_channels_, bool has_channels_>
struct cpw_params {
  static constexpr int num_channels = num_channels_;
  static constexpr bool has_channels = has_channels_;
};

using CyclicPixelWrapperValues =
    ::testing::Types<cpw_params<1, true>, cpw_params<3, true>, cpw_params<1, false>>;

TYPED_TEST_P(CyclicPixelWrapperTest, FillAndCycle) {
  constexpr int size = 6;
  constexpr int num_channels = TypeParam::num_channels;
  constexpr bool has_channels = TypeParam::has_channels;
  int tmp_buffer[size * num_channels];  // NOLINT
  int input_buffer[size * num_channels];  // NOLINT
  for (int i = 0; i < size * num_channels; i++) {
    input_buffer[i] = i;
    tmp_buffer[i] = -1;
  }
  CyclicPixelWrapper<int, has_channels> cpw(tmp_buffer, size, num_channels);
  EXPECT_EQ(0, cpw.Size());
  for (int i = 0; i < size; i++) {
    cpw.PushPixel(input_buffer + i * num_channels);
    EXPECT_EQ(tmp_buffer + i * num_channels, cpw.GetPixelOffset(i));
    for (int c = 0; c < num_channels; c++) {
      EXPECT_EQ(input_buffer[i * num_channels + c], cpw.GetPixelOffset(i)[c]);
    }
  }
  for (int i = 0; i < size; i++) {
    cpw.PopPixel();
    cpw.PushPixel(input_buffer + i * num_channels);
    for (int j = 0; j < size; j++) {
      // we're starting at i + 1 as we did already one Pop & Push operation
      int element = (i + 1 + j) % size;
      EXPECT_EQ(tmp_buffer + element * num_channels, cpw.GetPixelOffset(j));
      for (int c = 0; c < num_channels; c++) {
        EXPECT_EQ(input_buffer[element * num_channels + c], cpw.GetPixelOffset(j)[c]);
      }
    }
  }
}

void baseline_dot(span<int> result, span<const int> input, span<const int> window, int in_offset) {
  int num_channels = result.size();
  int num_elements = window.size();
  ASSERT_EQ(input.size(), num_channels * num_elements);
  for (int c = 0; c < num_channels; c++) {
    result[c] = 0;
    for (int i = 0; i < num_elements; i++) {
      int in_elem = (i + in_offset) % num_elements;
      result[c] += window[i] * input[in_elem * num_channels + c];
    }
  }
}

TYPED_TEST_P(CyclicPixelWrapperTest, DotProduct) {
  constexpr int size = 6;
  constexpr int num_channels = TypeParam::num_channels;
  constexpr bool has_channels = TypeParam::has_channels;
  int tmp_buffer[size * num_channels];  // NOLINT
  int input_buffer[size * num_channels];  // NOLINT
  int window[size];  // NOLINT
  for (int i = 0; i < size * num_channels; i++) {
    input_buffer[i] = i;
    tmp_buffer[i] = -1;
  }
  for (int i = 0; i < size; i++) {
    window[i] = i;
  }
  int baseline[num_channels], result[num_channels];
  for (int c = 0; c < num_channels; c++) {
    baseline[c] = 0;
    for (int i = 0; i < size; i++) {
      baseline[c] += window[i] * input_buffer[i * num_channels + c];
    }
  }

  CyclicPixelWrapper<int, has_channels> cpw(tmp_buffer, size, num_channels);
  for (int i = 0; i < size; i++) {
    cpw.PushPixel(input_buffer + i * num_channels);
  }
  cpw.CalculateDot(result, window);
  baseline_dot(make_span(baseline), make_span(input_buffer), make_span(window), 0);
  for (int c = 0; c < num_channels; c++) {
    EXPECT_EQ(baseline[c], result[c]);
  }
  for (int i = 0; i < size; i++) {
    cpw.PopPixel();
    cpw.PushPixel(input_buffer + i * num_channels);
    cpw.CalculateDot(result, window);
    // again we start here at i + 1 offset
    baseline_dot(make_span(baseline), make_span(input_buffer), make_span(window), i + 1);
    for (int c = 0; c < num_channels; c++) {
      EXPECT_EQ(baseline[c], result[c]);
    }
  }
}

REGISTER_TYPED_TEST_SUITE_P(CyclicPixelWrapperTest, FillAndCycle, DotProduct);

INSTANTIATE_TYPED_TEST_SUITE_P(CyclicPixelWrapper, CyclicPixelWrapperTest,
                               CyclicPixelWrapperValues);

template <typename Out, typename In, typename W>
void BaselineConvolveAxis(Out *out, const In *in, const W *window, int len, int r, int channel_num,
                          int64_t stride) {
  for (int i = 0; i < len; i++) {
    for (int c = 0; c < channel_num; c++) {
      out[i * stride + c] = 0;
      for (int d = -r; d <= r; d++) {
        if (i + d >= 0 && i + d < len) {
          out[i * stride + c] += in[(i + d) * stride + c] * window[d + r];
        } else {
          out[i * stride + c] +=
              in[boundary::idx_reflect_101(i + d, len) * stride + c] * window[d + r];
        }
      }
    }
  }
}

template <typename Out, typename In, typename W, int ndim>
void BaselineConvolve(const TensorView<StorageCPU, Out, ndim> &out,
                      const TensorView<StorageCPU, In, ndim> &in,
                      const TensorView<StorageCPU, W, 1> &window, int axis, int r,
                      int current_axis = 0, int64_t offset = 0) {
  if (current_axis == ndim - 1) {
    auto stride = GetStrides(out.shape)[axis];
    BaselineConvolveAxis(out.data + offset, in.data + offset, window.data, out.shape[axis], r,
                         in.shape[ndim - 1], stride);
  } else if (current_axis == axis) {
    BaselineConvolve(out, in, window, axis, r, current_axis + 1, offset);
  } else {
    for (int i = 0; i < out.shape[current_axis]; i++) {
      auto stride = GetStrides(out.shape)[current_axis];
      BaselineConvolve(out, in, window, axis, r, current_axis + 1, offset + i * stride);
    }
  }
}

template <int ndim_, bool has_channels_, int axis_, int window_size_>
struct convolution_params {
  static constexpr int ndim = ndim_;
  static constexpr int baseline_ndim = ndim + (has_channels_ ? 0 : 1);
  static constexpr bool has_channels = has_channels_;
  static constexpr int axis = axis_;
  static constexpr int window_size = window_size_;
};

template <typename T>
struct ConvolutionCpuKernelTest : public ::testing::Test {
  using Kernel = ConvolutionCpu<float, uint8_t, float, T::ndim, T::axis, T::has_channels>;

  TensorShape<T::ndim> GetShape() {
    if (T::has_channels) {
      return shape_ch_.template last<T::ndim>();
    } else {
      return shape_noch_.template last<T::ndim>();
    }
  }

  TensorShape<T::baseline_ndim> GetBaselineShape() {
    if (T::has_channels) {
      return shape_ch_.template last<T::baseline_ndim>();
    } else {
      return shape_noch1_.template last<T::baseline_ndim>();
    }
  }

  void SetUp() override {
    constexpr int window_size = T::window_size;
    kernel_window_.reshape(uniform_list_shape<1>(1, {window_size}));
    k_win_ = kernel_window_.cpu()[0];

    // almost box filter, with raised center
    for (int i = 0; i < window_size; i++) {
      if (i < window_size / 2) {
        k_win_.data[i] = 1;
      } else if (i == window_size / 2) {
        k_win_.data[i] = 2;
      } else {
        k_win_.data[i] = 1;
      }
    }

    input_.reshape(uniform_list_shape<T::ndim>(1, GetShape()));
    in_ = input_.cpu()[0];
    baseline_in_ = {in_.data, GetBaselineShape()};

    ConstantFill(in_, 0);

    std::mt19937 rng;
    UniformRandomFill(in_, rng, 0, 255);

    output_.reshape(uniform_list_shape<T::ndim>(1, GetShape()));
    baseline_output_.reshape(uniform_list_shape<T::baseline_ndim>(1, GetBaselineShape()));
    out_ = output_.cpu()[0];
    baseline_out_ = baseline_output_.cpu()[0];

    ConstantFill(out_, -1);
    ConstantFill(baseline_out_, -1);
  }

  void RunTest() {
    KernelContext ctx;
    Kernel kernel;

    auto req = kernel.Setup(ctx, in_, k_win_);
    // this is painful
    ScratchpadAllocator scratch_alloc;
    scratch_alloc.Reserve(req.scratch_sizes);
    auto scratchpad = scratch_alloc.GetScratchpad();
    ctx.scratchpad = &scratchpad;

    kernel.Run(ctx, out_, in_, k_win_);
    BaselineConvolve(baseline_out_, baseline_in_, k_win_, T::axis, T::window_size / 2);

    // for validation we need the same shape
    TensorView<StorageCPU, float, T::ndim> baseline_out_reshaped = {baseline_out_.data, out_.shape};
    Check(out_, baseline_out_reshaped);
  }

  TestTensorList<float, 1> kernel_window_;
  TestTensorList<uint8_t, T::ndim> input_;
  TestTensorList<float, T::ndim> output_;
  TestTensorList<float, T::baseline_ndim> baseline_output_;

  TensorView<StorageCPU, float, 1> k_win_;
  TensorView<StorageCPU, uint8_t, T::ndim> in_;
  TensorView<StorageCPU, uint8_t, T::baseline_ndim> baseline_in_;
  TensorView<StorageCPU, float, T::ndim> out_;
  TensorView<StorageCPU, float, T::baseline_ndim> baseline_out_;

  const TensorShape<> shape_ch_ = {13, 11, 21, 3};
  const TensorShape<> shape_noch_ = {13, 11, 21};
  const TensorShape<> shape_noch1_ = {13, 11, 21, 1};
};

TYPED_TEST_SUITE_P(ConvolutionCpuKernelTest);

using ConvolutionTestValues =
    ::testing::Types<convolution_params<1, false, 0, 3>, convolution_params<2, true, 0, 3>,
                     convolution_params<2, false, 0, 3>, convolution_params<2, false, 1, 3>,
                     convolution_params<3, true, 0, 3>, convolution_params<3, true, 1, 3>,
                     convolution_params<3, false, 1, 3>, convolution_params<3, false, 1, 7>,
                     convolution_params<3, false, 1, 11>, convolution_params<3, false, 1, 21>,
                     convolution_params<3, false, 1, 101>>;

TYPED_TEST_P(ConvolutionCpuKernelTest, DoConvolution) {
  this->RunTest();
}

REGISTER_TYPED_TEST_SUITE_P(ConvolutionCpuKernelTest, DoConvolution);
INSTANTIATE_TYPED_TEST_SUITE_P(ConvolutionCpuKernel, ConvolutionCpuKernelTest,
                               ConvolutionTestValues);

}  // namespace kernels
}  // namespace dali
