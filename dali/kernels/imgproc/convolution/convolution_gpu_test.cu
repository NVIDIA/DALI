// Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/kernels/imgproc/convolution/baseline_convolution.h"
#include "dali/kernels/imgproc/convolution/convolution_cpu.h"
#include "dali/kernels/imgproc/convolution/convolution_gpu.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
#include "dali/kernels/dynamic_scratchpad.h"

namespace dali {
namespace kernels {

enum transformCase {
  Scale05,
  AccScaleOutput
};

template <int transform_case, typename OutType, typename WinType, int ndim>
struct TransformParam;

template <typename OutType, typename WinType, int ndim>
struct TransformParam<Scale05, OutType, WinType, ndim> {
  using TransformCpu = conv_transform::TransScaleSat<OutType, WinType>;
  static constexpr transformCase transform_case = Scale05;
  explicit TransformParam(int nsamples) : nsamples_{nsamples} {}

  void FillOut(const TensorListView<StorageCPU, OutType, ndim> &out) {}

  TransformCpu GetCpuTransform(int sample_idx) {
    (void)sample_idx;
    return {0.5f};
  }

  ConvEpilogue GetGpuEpilogue() {
    return {0.5f};
  }

 private:
  int nsamples_;
};

template <typename OutType, typename WinType, int ndim>
struct TransformParam<AccScaleOutput, OutType, WinType, ndim> {
  using TransformCpu = conv_transform::TransScaleAddOutSat<OutType, WinType>;
  static constexpr transformCase transform_case = AccScaleOutput;
  explicit TransformParam(int nsamples) : nsamples_{nsamples}, scales_(nsamples) {
    for (int sample_idx = 0; sample_idx < nsamples_; sample_idx++) {
      scales_[sample_idx] = 1.f / (sample_idx + 1);
    }
  }

  void FillOut(const TensorListView<StorageCPU, OutType, ndim> &out) {
    assert(out.num_samples() == nsamples_);
    for (int sample_idx = 0; sample_idx < nsamples_; sample_idx++) {
      auto out_sample = out[sample_idx];
      int vol = volume(out_sample.shape);
      for (int i = 0; i < vol; i++) {
        out_sample.data[i] = i % 199;
      }
    }
  }

  TransformCpu GetCpuTransform(int sample_idx) {
    return {scales_[sample_idx]};
  }

  ConvEpilogue GetGpuEpilogue() {
    return {make_span(scales_), 1.f};
  }

 private:
  int nsamples_;
  std::vector<float> scales_;
};

template <int ndim_, bool has_channels_, int axis_, typename InType_, typename OutType_ = float,
          typename WinType_ = float, int transform_case_ = Scale05>
struct convolution_params {
  static constexpr int ndim = ndim_;
  static constexpr bool has_channels = has_channels_;
  static constexpr int axis = axis_;
  using InType = InType_;
  using WinType = WinType_;
  using OutType = OutType_;
  using TransformCase = TransformParam<transform_case_, OutType, WinType, ndim>;
};

template <typename T>
struct ConvolutionGpuKernelTest : public ::testing::Test {
  using InType = typename T::InType;
  using WinType = typename T::WinType;
  using OutType = typename T::OutType;
  using TransformCase = typename T::TransformCase;
  using TransformCpu = typename TransformCase::TransformCpu;
  using KernelCpu =
      ConvolutionCpu<OutType, InType, WinType, T::ndim, T::axis, T::has_channels, TransformCpu>;
  using KernelGpu = ConvolutionGpu<OutType, InType, WinType, T::ndim, T::axis, T::has_channels>;

  static constexpr transformCase transform_case = TransformCase::transform_case;

  TensorListShape<T::ndim> GetShape() {
    if (T::has_channels) {
      return shape_ch_.template last<T::ndim>();
    } else {
      return shape_noch_.template last<T::ndim>();
    }
  }

  void SetUp() override {
    kernel_window_.reshape(shape_window);
    k_win_ = kernel_window_.cpu();

    // almost box filter, with raised center
    for (int sample = 0; sample < shape_window.num_samples(); sample++) {
      int window_size = shape_window[sample][0];
      for (int i = 0; i < window_size; i++) {
        if (i < window_size / 2) {
          k_win_[sample].data[i] = 2;
        } else if (i == window_size / 2) {
          k_win_[sample].data[i] = 4;
        } else {
          k_win_[sample].data[i] = 2;
        }
      }
    }

    input_.reshape(GetShape());
    baseline_in_ = input_.cpu();

    std::mt19937 rng;
    UniformRandomFill(baseline_in_, rng, 0, 64);
    in_ = input_.gpu();

    output_.reshape(GetShape());
    baseline_output_.reshape(GetShape());
  }

  void RunTest() {
    KernelContext ctx_cpu, ctx_gpu;
    ctx_gpu.gpu.stream = 0;

    KernelCpu kernel_cpu;
    KernelGpu kernel_gpu;

    auto data_shape = GetShape();
    int num_samples = data_shape.size();

    TransformCase transform(num_samples);
    baseline_out_ = baseline_output_.cpu();
    transform.FillOut(baseline_out_);
    transform.FillOut(output_.cpu());
    out_ = output_.gpu();

    for (int sample = 0; sample < num_samples; sample++) {
      int window_size = shape_window[sample][0];
      auto req = kernel_cpu.Setup(ctx_cpu, data_shape[sample], window_size);

      DynamicScratchpad dyn_scratchpad_cpu(AccessOrder::host());
      ctx_cpu.scratchpad = &dyn_scratchpad_cpu;

      kernel_cpu.Run(ctx_cpu, baseline_out_[sample], baseline_in_[sample], k_win_[sample],
                     transform.GetCpuTransform(sample));
    }

    auto req = kernel_gpu.Setup(ctx_gpu, in_.shape, shape_window);

    auto gpu_epilogue = transform.GetGpuEpilogue();

    DynamicScratchpad dyn_scratchpad_gpu(AccessOrder(ctx_gpu.gpu.stream));
    ctx_gpu.scratchpad = &dyn_scratchpad_gpu;

    kernel_gpu.Run(ctx_gpu, out_, in_, k_win_, span<const int>{}, gpu_epilogue);

    output_.invalidate_cpu();
    auto out_cpu_ = output_.cpu();

    double eps = std::is_integral<OutType>::value && transform_case == AccScaleOutput ? 1 : 0.01;
    Check(out_cpu_, baseline_out_, EqualEps(eps));
  }

  TestTensorList<WinType, 1> kernel_window_;
  TestTensorList<InType, T::ndim> input_;
  TestTensorList<OutType, T::ndim> output_;
  TestTensorList<OutType, T::ndim> baseline_output_;

  TensorListView<StorageCPU, WinType, 1> k_win_;
  TensorListView<StorageGPU, InType, T::ndim> in_;
  TensorListView<StorageGPU, OutType, T::ndim> out_;
  TensorListView<StorageCPU, InType, T::ndim> baseline_in_;
  TensorListView<StorageCPU, OutType, T::ndim> baseline_out_;

  const TensorListShape<> shape_ch_ = {
      {29, 145, 128, 3}, {64, 64, 64, 3}, {164, 164, 164, 3}, {12, 12, 12, 3},   {4, 200, 180, 3},
      {200, 4, 180, 3},  {75, 75, 75, 5}, {16, 512, 512, 1},  {16, 512, 512, 1}, {16, 512, 512, 1},
      {8, 1, 32, 3},     {8, 32, 1, 3},   {1, 8, 32, 3}};
  const TensorListShape<> shape_noch_ = {
      {29, 145, 128}, {64, 64, 64}, {164, 164, 164}, {12, 12, 12},   {4, 200, 180},
      {200, 4, 180},  {75, 75, 75}, {16, 512, 512},  {16, 512, 512}, {16, 512, 512},
      {8, 1, 32},     {8, 32, 1},   {1, 8, 32}};
  // Three last window sizes are used to verify against off-by-one error when calculating
  // where the nonzero region ends in the kernel::Conv, as the ThreadblockShape::kK == 8.
  const TensorListShape<1> shape_window = {
      {1, 3, 5, 15, 25, 51, 101, 2 * 7 + 1, 2 * 8 + 1, 2 * 9 + 1, 7, 1, 11}};
};

TYPED_TEST_SUITE_P(ConvolutionGpuKernelTest);

// ndim, has_channels, convolution axis, input type, [output type = float]
using ConvolutionTestValues = ::testing::Types<
    // 1D
    convolution_params<1, false, 0, uint8_t, float>,
    convolution_params<1, false, 0, uint8_t, float, float, AccScaleOutput>,
    convolution_params<1, false, 0, int16_t, int16_t>,
    convolution_params<1, false, 0, int16_t, int16_t, int16_t>,
    // TODO(klecki): cannot test this as it tries to use device __half for CPU kernel
    // convolution_params<1, false, 0, float16, float16, float16>,
    convolution_params<1, false, 0, float, float>,
    convolution_params<1, false, 0, float, float, float, AccScaleOutput>,
    convolution_params<1, false, 0, int32_t, int32_t>,
    convolution_params<1, false, 0, int32_t, int32_t, float, AccScaleOutput>,
    convolution_params<1, false, 0, uint32_t, uint32_t>,
    convolution_params<1, false, 0, int16_t, int16_t>,
    convolution_params<1, false, 0, int16_t, int16_t, float, AccScaleOutput>,
    convolution_params<1, false, 0, int16_t, float>,
    convolution_params<1, false, 0, int16_t, float, float, AccScaleOutput>,
    convolution_params<1, false, 0, uint32_t, uint32_t>,
    // 1D with channels
    convolution_params<2, true, 0, float>,
    convolution_params<2, true, 0, float, float, float, AccScaleOutput>,
    // 2D outer, double
    convolution_params<2, false, 0, float>,
    convolution_params<2, false, 0, float, float, float, AccScaleOutput>,
    // 2D inner
    convolution_params<2, false, 1, float>,
    convolution_params<2, false, 1, float, float, float, AccScaleOutput>,
    // 2D outer with channels
    convolution_params<3, true, 0, float>,
    convolution_params<3, true, 0, float, float, float, AccScaleOutput>,
    // 2D inner with channels
    convolution_params<3, true, 1, float>,
    convolution_params<3, true, 1, float, float, float, AccScaleOutput>,
    // 3D outer
    convolution_params<3, false, 0, float>,
    convolution_params<3, false, 0, float, float, float, AccScaleOutput>,
    // 3D "middle"
    convolution_params<3, false, 1, float>,
    convolution_params<3, false, 1, float, float, float, AccScaleOutput>,
    // 3D inner
    convolution_params<3, false, 2, float>,
    convolution_params<3, false, 2, float, float, float, AccScaleOutput>,
    // 3D outer with channels
    convolution_params<4, true, 0, float>,
    convolution_params<4, true, 0, float, float, float, AccScaleOutput>,
    // 3D "middle" with channels
    convolution_params<4, true, 1, float>,
    convolution_params<4, true, 1, float, float, float, AccScaleOutput>,
    // 3D outer with channels
    convolution_params<4, true, 2, float>,
    convolution_params<4, true, 2, float, float, float, AccScaleOutput>>;

TYPED_TEST_P(ConvolutionGpuKernelTest, DoConvolution) {
  this->RunTest();
}

REGISTER_TYPED_TEST_SUITE_P(ConvolutionGpuKernelTest, DoConvolution);
INSTANTIATE_TYPED_TEST_SUITE_P(ConvolutionGpuKernel, ConvolutionGpuKernelTest,
                               ConvolutionTestValues);

}  // namespace kernels
}  // namespace dali
