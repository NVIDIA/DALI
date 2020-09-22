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
#include "dali/core/convert.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/imgproc/convolution/convolution_cpu.h"
#include "dali/kernels/scratch.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
#include "dali/kernels/imgproc/convolution/baseline_convolution.h"

namespace dali {
namespace kernels {

template <typename T>
struct CyclicWindowWrapperTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(CyclicWindowWrapperTest);

template <int num_lanes_>
struct cww_params {
  static constexpr int num_lanes = num_lanes_;
};

using CyclicWindowWrapperValues = ::testing::Types<cww_params<1>, cww_params<3>, cww_params<16>>;

TYPED_TEST_P(CyclicWindowWrapperTest, FillAndCycle) {
  constexpr int size = 6;
  constexpr int num_lanes = TypeParam::num_lanes;
  int tmp_buffer[size * num_lanes];    // NOLINT
  int input_buffer[size * num_lanes];  // NOLINT
  for (int i = 0; i < size * num_lanes; i++) {
    input_buffer[i] = i;
    tmp_buffer[i] = -1;
  }
  CyclicWindowWrapper<int, 16> cww(tmp_buffer, size, num_lanes);
  EXPECT_EQ(0, cww.Size());
  for (int i = 0; i < size; i++) {
    cww.PushBack(input_buffer + i * num_lanes);
    EXPECT_EQ(tmp_buffer + i * num_lanes, cww.GetElementOffset(i));
    for (int c = 0; c < num_lanes; c++) {
      EXPECT_EQ(input_buffer[i * num_lanes + c], cww.GetElementOffset(i)[c]);
    }
  }
  for (int i = 0; i < size; i++) {
    cww.PopFront();
    cww.PushBack(input_buffer + i * num_lanes);
    for (int j = 0; j < size; j++) {
      // we're starting at i + 1 as we did already one Pop & Push operation
      int element = (i + 1 + j) % size;
      EXPECT_EQ(tmp_buffer + element * num_lanes, cww.GetElementOffset(j));
      for (int c = 0; c < num_lanes; c++) {
        EXPECT_EQ(input_buffer[element * num_lanes + c], cww.GetElementOffset(j)[c]);
      }
    }
  }
}

void BaselineDot(span<int> result, span<const int> input, span<const int> window, int in_offset) {
  int num_lanes = result.size();
  int num_elements = window.size();
  ASSERT_EQ(input.size(), num_lanes * num_elements);
  for (int c = 0; c < num_lanes; c++) {
    result[c] = 0;
    for (int i = 0; i < num_elements; i++) {
      int in_elem = (i + in_offset) % num_elements;
      result[c] += window[i] * input[in_elem * num_lanes + c];
    }
  }
}

TYPED_TEST_P(CyclicWindowWrapperTest, DotProduct) {
  constexpr int size = 6;
  constexpr int num_lanes = TypeParam::num_lanes;
  int tmp_buffer[size * num_lanes];    // NOLINT
  int input_buffer[size * num_lanes];  // NOLINT
  int window[size];                    // NOLINT
  for (int i = 0; i < size * num_lanes; i++) {
    input_buffer[i] = i;
    tmp_buffer[i] = -1;
  }
  for (int i = 0; i < size; i++) {
    window[i] = i;
  }
  int baseline[num_lanes], result[num_lanes];
  for (int c = 0; c < num_lanes; c++) {
    baseline[c] = 0;
    result[c] = 0;
    for (int i = 0; i < size; i++) {
      baseline[c] += window[i] * input_buffer[i * num_lanes + c];
    }
  }

  CyclicWindowWrapper<int, 16> cww(tmp_buffer, size, num_lanes);
  for (int i = 0; i < size; i++) {
    cww.PushBack(input_buffer + i * num_lanes);
  }
  cww.CalculateDot(result, window);
  BaselineDot(make_span(baseline), make_span(input_buffer), make_span(window), 0);
  for (int c = 0; c < num_lanes; c++) {
    EXPECT_EQ(baseline[c], result[c]);
  }
  for (int i = 0; i < size; i++) {
    cww.PopFront();
    cww.PushBack(input_buffer + i * num_lanes);
    cww.CalculateDot(result, window);
    // again we start here at i + 1 offset
    BaselineDot(make_span(baseline), make_span(input_buffer), make_span(window), i + 1);
    for (int c = 0; c < num_lanes; c++) {
      EXPECT_EQ(baseline[c], result[c]);
    }
  }
}

REGISTER_TYPED_TEST_SUITE_P(CyclicWindowWrapperTest, FillAndCycle, DotProduct);

INSTANTIATE_TYPED_TEST_SUITE_P(CyclicWindowWrapper, CyclicWindowWrapperTest,
                               CyclicWindowWrapperValues);

template <int ndim_, bool has_channels_, int axis_, int window_size_, typename InType_,
          bool in_place_>
struct convolution_params {
  static constexpr int ndim = ndim_;
  static constexpr int baseline_ndim = ndim + (has_channels_ ? 0 : 1);
  static constexpr bool has_channels = has_channels_;
  static constexpr int axis = axis_;
  static constexpr int window_size = window_size_;
  static constexpr bool in_place = in_place_;
  using InType = InType_;
  static_assert(!in_place_ || std::is_same<InType, float>::value,
                "Input type must be float if you want to test in place transformation.");
};

template <typename T>
struct ConvolutionCpuKernelTest : public ::testing::Test {
  using Kernel =
      ConvolutionCpu<float, typename T::InType, float, T::ndim, T::axis, T::has_channels>;

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

    ConstantFill(in_, 0);

    std::mt19937 rng;
    UniformRandomFill(in_, rng, 0, 255);

    output_.reshape(uniform_list_shape<T::ndim>(1, GetShape()));
    if (T::in_place) {
      out_ = {reinterpret_cast<float *>(in_.data), in_.shape};  // so the compiler doesn't complain
    } else {
      out_ = output_.cpu()[0];
      ConstantFill(out_, -1);
    }
    baseline_input_.reshape(uniform_list_shape<T::baseline_ndim>(1, GetBaselineShape()));
    baseline_output_.reshape(uniform_list_shape<T::baseline_ndim>(1, GetBaselineShape()));
    baseline_in_ = baseline_input_.cpu()[0];
    baseline_out_ = baseline_output_.cpu()[0];
    memcpy(baseline_in_.data, in_.data, volume(in_.shape) * sizeof(typename T::InType));

    ConstantFill(baseline_out_, -1);
  }

  void RunTest() {
    KernelContext ctx;
    Kernel kernel;

    auto req = kernel.Setup(ctx, in_.shape, k_win_.num_elements());
    // this is painful
    ScratchpadAllocator scratch_alloc;
    scratch_alloc.Reserve(req.scratch_sizes);
    auto scratchpad = scratch_alloc.GetScratchpad();
    ctx.scratchpad = &scratchpad;

    testing::BaselineConvolve(baseline_out_, baseline_in_, k_win_, T::axis, T::window_size / 2);
    kernel.Run(ctx, out_, in_, k_win_);

    // for validation we need the same shape
    TensorView<StorageCPU, float, T::ndim> baseline_out_reshaped = {baseline_out_.data, out_.shape};
    Check(out_, baseline_out_reshaped);
  }

  TestTensorList<float, 1> kernel_window_;
  TestTensorList<typename T::InType, T::ndim> input_;
  TestTensorList<typename T::InType, T::baseline_ndim> baseline_input_;
  TestTensorList<float, T::ndim> output_;
  TestTensorList<float, T::baseline_ndim> baseline_output_;

  TensorView<StorageCPU, float, 1> k_win_;
  TensorView<StorageCPU, typename T::InType, T::ndim> in_;
  TensorView<StorageCPU, typename T::InType, T::baseline_ndim> baseline_in_;
  TensorView<StorageCPU, float, T::ndim> out_;
  TensorView<StorageCPU, float, T::baseline_ndim> baseline_out_;

  const TensorShape<> shape_ch_ = {13, 11, 21, 3};
  const TensorShape<> shape_noch_ = {13, 11, 21};
  const TensorShape<> shape_noch1_ = {13, 11, 21, 1};
};

TYPED_TEST_SUITE_P(ConvolutionCpuKernelTest);

using ConvolutionTestValues = ::testing::Types<convolution_params<1, false, 0, 1, uint8_t, false>,
                                               convolution_params<1, false, 0, 3, uint8_t, false>,
                                               convolution_params<1, false, 0, 21, uint8_t, false>,
                                               convolution_params<1, false, 0, 51, uint8_t, false>,
                                               convolution_params<2, true, 0, 1, uint8_t, false>,
                                               convolution_params<2, true, 0, 3, uint8_t, false>,
                                               convolution_params<2, true, 0, 21, uint8_t, false>,
                                               convolution_params<2, true, 0, 51, uint8_t, false>,

                                               convolution_params<1, false, 0, 1, float, true>,
                                               convolution_params<1, false, 0, 3, float, true>,
                                               convolution_params<1, false, 0, 21, float, true>,
                                               convolution_params<1, false, 0, 51, float, true>,
                                               convolution_params<2, true, 0, 1, float, true>,
                                               convolution_params<2, true, 0, 3, float, true>,
                                               convolution_params<2, true, 0, 21, float, true>,
                                               convolution_params<2, true, 0, 51, float, true>,

                                               convolution_params<2, false, 0, 1, uint8_t, false>,
                                               convolution_params<2, false, 0, 3, uint8_t, false>,
                                               convolution_params<2, false, 1, 1, uint8_t, false>,
                                               convolution_params<2, false, 1, 3, uint8_t, false>,
                                               convolution_params<3, true, 0, 3, uint8_t, false>,
                                               convolution_params<3, true, 1, 3, uint8_t, false>,

                                               convolution_params<3, false, 1, 1, uint8_t, false>,
                                               convolution_params<3, false, 1, 3, uint8_t, false>,
                                               convolution_params<3, false, 1, 7, uint8_t, false>,
                                               convolution_params<3, false, 1, 11, uint8_t, false>,
                                               convolution_params<3, false, 1, 21, uint8_t, false>,
                                               convolution_params<3, false, 1, 101, uint8_t, false>,

                                               convolution_params<3, false, 1, 1, float, true>,
                                               convolution_params<3, false, 1, 3, float, true>,
                                               convolution_params<3, false, 1, 21, float, true>,
                                               convolution_params<3, false, 1, 101, float, true>>;

TYPED_TEST_P(ConvolutionCpuKernelTest, DoConvolution) {
  this->RunTest();
}

REGISTER_TYPED_TEST_SUITE_P(ConvolutionCpuKernelTest, DoConvolution);
INSTANTIATE_TYPED_TEST_SUITE_P(ConvolutionCpuKernel, ConvolutionCpuKernelTest,
                               ConvolutionTestValues);

}  // namespace kernels
}  // namespace dali
