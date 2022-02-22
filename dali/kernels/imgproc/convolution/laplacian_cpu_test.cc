// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <array>

#include "dali/kernels/common/utils.h"
#include "dali/kernels/imgproc/convolution/laplacian_cpu.h"
#include "dali/kernels/imgproc/convolution/laplacian_test.h"
#include "dali/kernels/scratch.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace kernels {

template <typename Out, int axes>
inline std::enable_if_t<std::is_integral<Out>::value> CompareOut(
    TensorView<StorageCPU, Out, axes>& out, TensorView<StorageCPU, Out, axes>& baseline) {
  Check(out, baseline);
}

template <typename Out, int axes>
inline std::enable_if_t<!std::is_integral<Out>::value> CompareOut(
    TensorView<StorageCPU, Out, axes>& out, TensorView<StorageCPU, Out, axes>& baseline) {
  Check(out, baseline, EqualEpsRel(1e-6, 1e-5));
}

void FillSobelWindow(span<float> window, int d_order) {
  int window_size = window.size();
  window[0] = 1.;
  for (int i = 1; i < window_size - d_order; i++) {
    auto prevval = window[0];
    for (int j = 1; j < i; j++) {
      auto val = window[j];
      window[j] = prevval + window[j];
      prevval = val;
    }
    window[i] = prevval;
  }
  for (int i = window_size - d_order; i < window_size; i++) {
    auto prevval = window[0];
    window[0] = -prevval;
    for (int j = 1; j < i; j++) {
      auto val = window[j];
      window[j] = prevval - window[j];
      prevval = val;
    }
    window[i] = prevval;
  }
}

template <int axes, int window_size>
void FillKernel(const TensorView<StorageCPU, float, axes>& kernel,
                const std::array<std::array<std::array<float, window_size>, axes>, axes>& windows,
                int padding) {
  for (int d = 0; d < axes; d++) {
    for (int i = 0; i < volume(kernel.shape); i++) {
      int offset = i;
      float r = 1.f;
      for (int j = axes - 1; j >= 0; j--) {
        int ind = offset % (window_size + padding);
        ind -= padding / 2;
        r *= (ind < 0 || ind >= window_size) ? 0.f : windows[d][j][ind];
        offset /= (window_size + padding);
      }
      kernel.data[i] += r;
    }
  }
}

template <int axes_, int window_size_, bool use_smoothing_>
struct LaplacianWindows {
  static constexpr int axes = axes_;
  static constexpr int window_size = window_size_;
  static constexpr bool use_smoothing = use_smoothing_;

  LaplacianWindows() {
    for (int i = 0; i < axes; i++) {
      for (int j = 0; j < axes; j++) {
        if (i == j) {
          window_sizes[i][j] = window_size;
          FillSobelWindow(make_span(windows[i][j]), 2);
          tensor_windows[i][j] = {windows[i][j].data(), window_size};
        } else if (use_smoothing) {
          window_sizes[i][j] = window_size;
          FillSobelWindow(make_span(windows[i][j]), 0);
          tensor_windows[i][j] = {windows[i][j].data(), window_size};
        } else {
          window_sizes[i][j] = 1;
          windows[i][j] = uniform_array<window_size>(0.f);
          auto middle = window_size / 2;
          windows[i][j][middle] = 1.f;
          tensor_windows[i][j] = {windows[i][j].data() + middle, 1};
        }
      }
    }
  }
  std::array<std::array<int, axes>, axes> window_sizes;
  std::array<std::array<std::array<float, window_size>, axes>, axes> windows;
  std::array<std::array<TensorView<StorageCPU, const float, 1>, axes>, axes> tensor_windows;
};

/**
 * @brief ``ndim`` laplacian is sum of ``ndim`` separable convolutions, each consiting of ``ndim``
 * one dimensional convolutions. It is equivalent to computing ``ndim`` convolution with a kernel
 * being the sum of ``ndim`` outer products of ``ndim`` 1 dimensional kernels. This test suite
 * computes that kernel and checks if it matches the laplacian computed on the tensor consising
 * of a single 1 surrounded with zeros.
 */
template <typename T>
struct LaplacianCpuKernelTest : public ::testing::Test {
  static constexpr int axes = T::axes;
  static constexpr int window_size = T::window_size;
  using Kernel = LaplacianCpu<float, float, float, axes, false>;

  void SetUp() override {
    int padding = 2;  // add padding zeros so that border101 has no effect
    auto dims = uniform_array<axes>(window_size + padding);
    TensorListShape<axes> kernel_shape = uniform_list_shape<axes>(3, dims);
    tensor_list_.reshape(kernel_shape);
    kernel_ = tensor_list_.cpu()[0];
    in_ = tensor_list_.cpu()[1];
    out_ = tensor_list_.cpu()[2];
    ConstantFill(kernel_, 0);
    FillKernel<axes, window_size>(kernel_, lapl_params_.windows, padding);
    ConstantFill(in_, 0);
    in_.data[volume(in_.shape) / 2] = 1.f;
  }

  void RunTest() {
    Kernel kernel;
    KernelContext ctx = {};

    auto req = kernel.Setup(ctx, in_.shape, lapl_params_.window_sizes);

    ScratchpadAllocator scratch_alloc;
    scratch_alloc.Reserve(req.scratch_sizes);
    auto scratchpad = scratch_alloc.GetScratchpad();
    ctx.scratchpad = &scratchpad;

    kernel.Run(ctx, out_, in_, lapl_params_.tensor_windows, uniform_array<axes>(1.f));
    CompareOut(out_, kernel_);
  }

  T lapl_params_;
  TestTensorList<float, axes> tensor_list_;
  TensorView<StorageCPU, float, axes> kernel_;
  TensorView<StorageCPU, float, axes> in_;
  TensorView<StorageCPU, float, axes> out_;
};


TYPED_TEST_SUITE_P(LaplacianCpuKernelTest);

using LaplacianKernelTestValues = ::testing::Types<
    LaplacianWindows<1, 3, false>,
    LaplacianWindows<1, 5, false>,
    LaplacianWindows<1, 7, false>,
    LaplacianWindows<1, 9, false>,
    LaplacianWindows<1, 11, false>,
    LaplacianWindows<1, 13, false>,
    LaplacianWindows<1, 15, false>,
    LaplacianWindows<1, 17, false>,
    LaplacianWindows<1, 19, false>,
    LaplacianWindows<1, 23, false>,

    LaplacianWindows<2, 3, true>,
    LaplacianWindows<2, 5, true>,
    LaplacianWindows<2, 7, true>,
    LaplacianWindows<2, 9, true>,
    LaplacianWindows<2, 11, true>,
    LaplacianWindows<2, 13, true>,
    LaplacianWindows<2, 15, true>,
    LaplacianWindows<2, 17, true>,
    LaplacianWindows<2, 19, true>,
    LaplacianWindows<2, 23, true>,

    LaplacianWindows<2, 3, false>,
    LaplacianWindows<2, 5, false>,
    LaplacianWindows<2, 7, false>,
    LaplacianWindows<2, 9, false>,
    LaplacianWindows<2, 11, false>,
    LaplacianWindows<2, 13, false>,
    LaplacianWindows<2, 15, false>,
    LaplacianWindows<2, 17, false>,
    LaplacianWindows<2, 19, false>,
    LaplacianWindows<2, 23, false>,

    LaplacianWindows<3, 3, true>,
    LaplacianWindows<3, 5, true>,
    LaplacianWindows<3, 7, true>,
    LaplacianWindows<3, 9, true>,
    LaplacianWindows<3, 11, true>,
    LaplacianWindows<3, 13, true>,
    LaplacianWindows<3, 15, true>,
    LaplacianWindows<3, 17, true>,
    LaplacianWindows<3, 19, true>,
    LaplacianWindows<3, 23, true>,

    LaplacianWindows<3, 3, false>,
    LaplacianWindows<3, 5, false>,
    LaplacianWindows<3, 7, false>,
    LaplacianWindows<3, 9, false>,
    LaplacianWindows<3, 11, false>,
    LaplacianWindows<3, 13, false>,
    LaplacianWindows<3, 15, false>,
    LaplacianWindows<3, 17, false>,
    LaplacianWindows<3, 19, false>,
    LaplacianWindows<3, 23, false>>;

TYPED_TEST_P(LaplacianCpuKernelTest, ExtractKernel) {
  this->RunTest();
}

REGISTER_TYPED_TEST_SUITE_P(LaplacianCpuKernelTest, ExtractKernel);
INSTANTIATE_TYPED_TEST_SUITE_P(LaplacianCpuKernel, LaplacianCpuKernelTest,
                               LaplacianKernelTestValues);

template <typename Out_, typename In_, int axes_, int window_size_, bool has_channels_,
          bool use_smoothing_>
struct test_laplacian {
  static constexpr int axes = axes_;
  static constexpr bool has_channels = has_channels_;
  static constexpr int ndim = has_channels ? axes + 1 : axes;
  static constexpr int window_size = window_size_;
  static constexpr bool use_smoothing = use_smoothing_;
  using Out = Out_;
  using In = In_;
};

/**
 * @brief Computes laplacian and compares it against simple baseline implementation
 * that accumulates all the separable convolutions in a separate buffer.
 */
template <typename T>
struct LaplacianCpuTest : public ::testing::Test {
  static constexpr int axes = T::axes;
  static constexpr bool has_channels = T::has_channels;
  static constexpr int ndim = T::ndim;
  static constexpr int window_size = T::window_size;
  static constexpr bool use_smoothing = T::use_smoothing;
  static constexpr std::array<int, 3> dim_sizes = {41, 19, 37};
  static constexpr std::array<float, 3> axe_weights = {0.5f, 0.25f, 1.f};
  using Out = typename T::Out;
  using In = typename T::In;
  using W = float;
  using Kernel = LaplacianCpu<Out, In, W, axes, has_channels>;
  using Conv = SeparableConvolutionCpu<W, In, W, axes, has_channels>;

  static std::array<int, ndim> GetShape() {
    static_assert(ndim == axes + has_channels);
    std::array<int, ndim> shape;
    for (int i = 0; i < axes; i++) {
      shape[i] = dim_sizes[i];
    }
    if (has_channels) {
      shape[ndim - 1] = 3;
    }
    return shape;
  }

  static std::array<float, axes> GetWeights() {
    std::array<float, axes> weights;
    for (int i = 0; i < axes; i++) {
      weights[i] = axe_weights[i];
    }
    return weights;
  }

  LaplacianCpuTest() : shape_{GetShape()}, weights_{GetWeights()} {}

  void RunBaseline() {
    Conv kernel;
    KernelContext ctx = {};
    auto vol = volume(shape_);
    for (int axis = 0; axis < axes; axis++) {
      auto req = kernel.Setup(ctx, shape_, lapl_params_.window_sizes[axis]);
      ScratchpadAllocator scratch_alloc;
      scratch_alloc.Reserve(req.scratch_sizes);
      auto scratchpad = scratch_alloc.GetScratchpad();
      ctx.scratchpad = &scratchpad;
      kernel.Run(ctx, intermediate_, in_, lapl_params_.tensor_windows[axis]);
      for (int i = 0; i < vol; i++) {
        baseline_acc_.data[i] += weights_[axis] * intermediate_.data[i];
      }
    }
    for (int i = 0; i < vol; i++) {
      baseline_out_.data[i] = ConvertSat<Out>(baseline_acc_.data[i]);
    }
  }

  void SetUp() override {
    TensorListShape<ndim> in_data_shape = uniform_list_shape<ndim>(1, shape_);
    TensorListShape<ndim> out_data_shape = uniform_list_shape<ndim>(2, shape_);
    in_list_.reshape(in_data_shape);
    in_ = in_list_.cpu()[0];
    std::mt19937 rng;
    UniformRandomFill(in_, rng, 0, 255);
    out_list_.reshape(out_data_shape);
    out_ = out_list_.cpu()[0];
    ConstantFill(out_, -1);
    baseline_out_ = out_list_.cpu()[1];
    acc_list_.reshape(out_data_shape);
    intermediate_ = acc_list_.cpu()[0];
    baseline_acc_ = acc_list_.cpu()[1];
    ConstantFill(baseline_acc_, 0);
  }

  void RunTest() {
    Kernel kernel;
    KernelContext ctx = {};
    auto req = kernel.Setup(ctx, in_.shape, lapl_params_.window_sizes);
    ScratchpadAllocator scratch_alloc;
    scratch_alloc.Reserve(req.scratch_sizes);
    auto scratchpad = scratch_alloc.GetScratchpad();
    ctx.scratchpad = &scratchpad;

    kernel.Run(ctx, out_, in_, lapl_params_.tensor_windows, weights_);
    RunBaseline();
    CompareOut(out_, baseline_out_);
  }

  std::array<int, ndim> shape_;
  std::array<float, axes> weights_;
  LaplacianWindows<axes, window_size, use_smoothing> lapl_params_;
  TestTensorList<In, ndim> in_list_;
  TestTensorList<Out, ndim> out_list_;
  TestTensorList<float, ndim> acc_list_;
  TensorView<StorageCPU, In, ndim> in_;
  TensorView<StorageCPU, Out, ndim> out_;
  TensorView<StorageCPU, Out, ndim> baseline_out_;
  TensorView<StorageCPU, float, ndim> intermediate_;
  TensorView<StorageCPU, float, ndim> baseline_acc_;
};


TYPED_TEST_SUITE_P(LaplacianCpuTest);
using LaplacianTestValues = ::testing::Types<
    test_laplacian<float, float, 1, 3, true, false>,
    test_laplacian<float, float, 1, 7, true, false>,
    test_laplacian<float, float, 1, 11, true, false>,
    test_laplacian<float, float, 1, 23, true, false>,
    test_laplacian<uint8_t, uint8_t, 1, 3, true, false>,
    test_laplacian<uint8_t, uint8_t, 1, 7, true, false>,
    test_laplacian<uint8_t, uint8_t, 1, 11, true, false>,
    test_laplacian<uint8_t, uint8_t, 1, 23, true, false>,
    test_laplacian<float, float, 1, 3, false, false>,
    test_laplacian<float, float, 1, 7, false, false>,
    test_laplacian<float, float, 1, 11, false, false>,
    test_laplacian<float, float, 1, 23, false, false>,
    test_laplacian<uint8_t, uint8_t, 1, 3, false, false>,
    test_laplacian<uint8_t, uint8_t, 1, 7, false, false>,
    test_laplacian<uint8_t, uint8_t, 1, 11, false, false>,
    test_laplacian<uint8_t, uint8_t, 1, 23, false, false>,

    test_laplacian<float, float, 2, 3, true, true>,
    test_laplacian<float, float, 2, 5, true, true>,
    test_laplacian<float, float, 2, 19, true, true>,
    test_laplacian<float, float, 2, 23, true, true>,
    test_laplacian<uint8_t, uint8_t, 2, 3, true, true>,
    test_laplacian<uint8_t, uint8_t, 2, 5, true, true>,
    test_laplacian<uint8_t, uint8_t, 2, 7, true, true>,
    test_laplacian<uint8_t, uint8_t, 2, 23, true, true>,
    test_laplacian<float, float, 2, 3, true, false>,
    test_laplacian<float, float, 2, 7, true, false>,
    test_laplacian<float, float, 2, 11, true, false>,
    test_laplacian<float, float, 2, 23, true, false>,
    test_laplacian<uint8_t, uint8_t, 2, 3, true, false>,
    test_laplacian<uint8_t, uint8_t, 2, 7, true, false>,
    test_laplacian<uint8_t, uint8_t, 2, 11, true, false>,
    test_laplacian<uint8_t, uint8_t, 2, 23, true, false>,
    test_laplacian<float, float, 2, 3, false, true>,
    test_laplacian<float, float, 2, 5, false, true>,
    test_laplacian<float, float, 2, 19, false, true>,
    test_laplacian<float, float, 2, 23, false, true>,
    test_laplacian<uint8_t, uint8_t, 2, 3, false, true>,
    test_laplacian<uint8_t, uint8_t, 2, 5, false, true>,
    test_laplacian<uint8_t, uint8_t, 2, 7, false, true>,
    test_laplacian<uint8_t, uint8_t, 2, 23, false, true>,
    test_laplacian<float, float, 2, 3, false, false>,
    test_laplacian<float, float, 2, 7, false, false>,
    test_laplacian<float, float, 2, 11, false, false>,
    test_laplacian<float, float, 2, 23, false, false>,
    test_laplacian<uint8_t, uint8_t, 2, 3, false, false>,
    test_laplacian<uint8_t, uint8_t, 2, 7, false, false>,
    test_laplacian<uint8_t, uint8_t, 2, 11, false, false>,
    test_laplacian<uint8_t, uint8_t, 2, 23, false, false>,

    test_laplacian<float, float, 3, 3, true, true>,
    test_laplacian<float, float, 3, 5, true, true>,
    test_laplacian<float, float, 3, 19, true, true>,
    test_laplacian<float, float, 3, 23, true, true>,
    test_laplacian<uint8_t, uint8_t, 3, 3, true, true>,
    test_laplacian<uint8_t, uint8_t, 3, 5, true, true>,
    test_laplacian<uint8_t, uint8_t, 3, 7, true, true>,
    test_laplacian<uint8_t, uint8_t, 3, 23, true, true>,
    test_laplacian<float, float, 3, 3, true, false>,
    test_laplacian<float, float, 3, 7, true, false>,
    test_laplacian<float, float, 3, 11, true, false>,
    test_laplacian<float, float, 3, 23, true, false>,
    test_laplacian<uint8_t, uint8_t, 3, 3, true, false>,
    test_laplacian<uint8_t, uint8_t, 3, 7, true, false>,
    test_laplacian<uint8_t, uint8_t, 3, 11, true, false>,
    test_laplacian<uint8_t, uint8_t, 3, 23, true, false>,
    test_laplacian<float, float, 3, 3, false, true>,
    test_laplacian<float, float, 3, 5, false, true>,
    test_laplacian<float, float, 3, 19, false, true>,
    test_laplacian<float, float, 3, 23, false, true>,
    test_laplacian<uint8_t, uint8_t, 3, 3, false, true>,
    test_laplacian<uint8_t, uint8_t, 3, 5, false, true>,
    test_laplacian<uint8_t, uint8_t, 3, 7, false, true>,
    test_laplacian<uint8_t, uint8_t, 3, 23, false, true>,
    test_laplacian<float, float, 3, 3, false, false>,
    test_laplacian<float, float, 3, 7, false, false>,
    test_laplacian<float, float, 3, 11, false, false>,
    test_laplacian<float, float, 3, 23, false, false>,
    test_laplacian<uint8_t, uint8_t, 3, 3, false, false>,
    test_laplacian<uint8_t, uint8_t, 3, 7, false, false>,
    test_laplacian<uint8_t, uint8_t, 3, 11, false, false>,
    test_laplacian<uint8_t, uint8_t, 3, 23, false, false>>;

TYPED_TEST_P(LaplacianCpuTest, DoLaplacian) {
  this->RunTest();
}

REGISTER_TYPED_TEST_SUITE_P(LaplacianCpuTest, DoLaplacian);
INSTANTIATE_TYPED_TEST_SUITE_P(LaplacianCpuKernel, LaplacianCpuTest, LaplacianTestValues);

}  // namespace kernels
}  // namespace dali
