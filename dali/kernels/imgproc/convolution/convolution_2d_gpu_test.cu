// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/convert.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/kernels/imgproc/convolution/border_mode.h"
#include "dali/kernels/imgproc/convolution/convolution_2d_gpu.cuh"
#include "dali/kernels/scratch.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace kernels {

template <bool has_channels_, bool is_sequence_, typename InType_, typename OutType_,
          int filters_shift_, FilterBorderMode border_mode_>
struct ConvParams {
  static constexpr int axes = 2;
  static constexpr int filter_dim = axes;
  static constexpr bool has_channels = has_channels_;
  static constexpr bool is_sequence = is_sequence_;
  static constexpr int ndim = static_cast<int>(is_sequence) + axes + static_cast<int>(has_channels);
  static constexpr int filters_shift = filters_shift_;
  static constexpr FilterBorderMode border_mode = border_mode_;
  using InType = InType_;
  using OutType = OutType_;
  using WinType = float;
};

template <typename T>
struct Convolution2DGpuKernelTest : public ::testing::Test {
  using InType = typename T::InType;
  using WinType = typename T::WinType;
  using OutType = typename T::OutType;
  using Kernel = Convolution2dGpu<OutType, InType, WinType, T::has_channels, T::is_sequence>;

  TensorListShape<T::ndim> GetInputShape() {
    if (T::has_channels) {
      return shape_ch_.template last<T::ndim>();
    } else {
      return shape_noch_.template last<T::ndim>();
    }
  }

  TensorListShape<T::ndim> GetOutputShape() {
    return GetInputShape();
  }

  void SetUp() override {
    auto in_shape = GetInputShape();
    int num_samples = in_shape.num_samples();
    input_.reshape(in_shape);
    in_view_cpu_ = input_.cpu();
    std::mt19937 rng;
    UniformRandomFill(in_view_cpu_, rng, 0, 64);
    in_view_ = input_.gpu();
    output_.reshape(GetOutputShape());
    anchors_.reshape(uniform_list_shape<1>(num_samples, {T::filter_dim}));
    anchors_view_ = anchors_.cpu();
    filter_shapes_.resize(num_samples);
    int filter_idx = T::filters_shift;
    int num_base_filters = filter_shape_base_.num_samples();
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++, filter_idx++) {
      int idx = filter_idx % num_base_filters;
      TensorShape<2> shape = filter_shape_base_[idx];
      filter_shapes_.set_tensor_shape(sample_idx, shape);
      anchors_view_[sample_idx].data[0] = anchors_base[idx][0];
      anchors_view_[sample_idx].data[1] = anchors_base[idx][1];
    }
    filters_.reshape(filter_shapes_);
    filters_view_cpu_ = filters_.cpu();

    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      WinType w = 1;
      for (int x = 0; x < filter_shapes_[sample_idx][1]; x++) {
        for (int y = 0; y < filter_shapes_[sample_idx][0]; y++) {
          filters_view_cpu_[sample_idx].data[y * filter_shapes_[sample_idx][1] + x] = w;
          w += 1;
        }
      }
    }
    filters_view_ = filters_.gpu();
  }

  void RunTest() {
    KernelContext ctx_gpu;
    ctx_gpu.gpu.stream = 0;
    DynamicScratchpad dyn_scratchpad({}, AccessOrder(ctx_gpu.gpu.stream));
    ctx_gpu.scratchpad = &dyn_scratchpad;
    Kernel kernel_gpu;

    auto data_shape = GetInputShape();
    int num_samples = data_shape.size();

    baseline_out_ = baseline_output_.cpu();
    FillBaseline();
    out_view_ = output_.gpu();
    kernel_gpu.Run(ctx_gpu, out_view_, in_view_, filters_view_, anchors_view_, T::border_mode);
    auto out_cpu = output_.cpu();

    // double eps = std::is_integral<OutType>::value && transform_case == AccScaleOutput ? 1 : 0.01;
    // Check(out_cpu_, baseline_out_, EqualEps(eps));
  }

  void FillBaseline() {}

  TensorListShape<2> filter_shapes_;
  TestTensorList<WinType, T::filter_dim> filters_;
  TestTensorList<int, 1> anchors_;
  TestTensorList<InType, T::ndim> input_;
  TestTensorList<OutType, T::ndim> output_;
  TestTensorList<OutType, T::ndim> baseline_output_;

  TensorListView<StorageCPU, WinType, T::filter_dim> filters_view_cpu_;
  TensorListView<StorageGPU, WinType, T::filter_dim> filters_view_;
  TensorListView<StorageCPU, int, 1> anchors_view_;
  TensorListView<StorageCPU, InType, T::ndim> in_view_cpu_;
  TensorListView<StorageGPU, InType, T::ndim> in_view_;
  TensorListView<StorageGPU, OutType, T::ndim> out_view_;
  TensorListView<StorageCPU, OutType, T::ndim> baseline_out_;

  const TensorListShape<> shape_ch_ = {
      {29, 145, 128, 3}, {64, 64, 64, 3}, {164, 164, 164, 3}, {12, 12, 12, 3},   {4, 200, 180, 3},
      {200, 4, 180, 3},  {75, 75, 75, 5}, {16, 512, 512, 1},  {16, 512, 512, 1}, {16, 512, 512, 1},
      {8, 1, 32, 3},     {8, 32, 1, 3},   {1, 8, 32, 3},      {1, 111, 57, 129}, {1, 512, 512, 256},
      {16, 1, 517, 3},   {16, 517, 1, 3}};
  const TensorListShape<> shape_noch_ = {
      {29, 145, 128}, {64, 64, 64}, {164, 164, 164}, {12, 12, 12},   {4, 200, 180},
      {200, 4, 180},  {75, 75, 75}, {16, 512, 512},  {16, 512, 512}, {16, 512, 512},
      {8, 1, 32},     {8, 32, 1},   {1, 8, 32}};

  const TensorListShape<> filter_shape_base_ = {{3, 3},   {7, 7},   {101, 1},
                                                {1, 101}, {15, 17}, {4, 2}};
  const TensorListShape<> anchors_base = {{1, 1}, {3, 3}, {50, 0}, {0, 50}, {1, 16}, {2, 0}};
};

TYPED_TEST_SUITE_P(Convolution2DGpuKernelTest);

// ndim, has_channels, convolution axis, input type, [output type = float]
using ConvolutionTestValues =
    ::testing::Types<ConvParams<true, true, float, float, 0, DALI_BORDER_REFLECT_101>,
                     ConvParams<false, true, float, float, 1, DALI_BORDER_REFLECT_101>,
                     ConvParams<true, false, float, float, 2, DALI_BORDER_REFLECT_101>,
                     ConvParams<false, false, float, float, 3, DALI_BORDER_REFLECT_101>>;

TYPED_TEST_P(Convolution2DGpuKernelTest, DoConvolution) {
  this->RunTest();
}

REGISTER_TYPED_TEST_SUITE_P(Convolution2DGpuKernelTest, DoConvolution);
INSTANTIATE_TYPED_TEST_SUITE_P(Convolution2DGpuKernel, Convolution2DGpuKernelTest,
                               ConvolutionTestValues);

}  // namespace kernels
}  // namespace dali
