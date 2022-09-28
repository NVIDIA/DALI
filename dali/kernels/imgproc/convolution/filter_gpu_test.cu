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

#include "dali/core/boundary.h"
#include "dali/core/convert.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/kernels/imgproc/convolution/filter_gpu.cuh"
#include "dali/kernels/scratch.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace kernels {

using namespace boundary;  // NOLINT(build/namespaces)

template <bool is_sequence, bool has_channels, typename InT, typename OutT, typename WT, int ndim,
          typename Intermediate = decltype(std::declval<InT>() * std::declval<WT>())>
void baseline_conv(const TensorView<StorageCPU, InT, ndim> &in_view,
                   const TensorView<StorageCPU, OutT, ndim> &out_view,
                   const TensorView<StorageCPU, WT, 2> &filter_view,
                   const TensorView<StorageCPU, int, 1> &anchor_view) {
  int F = is_sequence ? in_view.shape[0] : 1;
  int H = in_view.shape[is_sequence];
  int W = in_view.shape[is_sequence + 1];
  int C = has_channels ? in_view.shape[is_sequence + 2] : 1;
  int R = filter_view.shape[0];
  int S = filter_view.shape[1];
  int anchor_r = -anchor_view.data[0];
  int anchor_s = -anchor_view.data[1];
  auto *filter_data = filter_view.data;
  for (int f = 0; f < F; f++) {
    const auto *in_data = in_view.data + f * H * W * C;
    auto *out_data = out_view.data + f * H * W * C;
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        for (int c = 0; c < C; c++) {
          Intermediate acc = 0;
          for (int r = 0; r < R; r++) {
            for (int s = 0; s < S; s++) {
              int y_idx = y + r + anchor_r;
              int x_idx = x + s + anchor_s;
              y_idx = idx_reflect_101(y_idx, 0, H);
              x_idx = idx_reflect_101(x_idx, 0, W);
              acc += in_data[y_idx * W * C + x_idx * C + c] * filter_data[r * S + s];
            }
          }
          out_data[y * W * C + x * C + c] = ConvertSat<OutT>(acc);
        }
      }
    }
  }
}

template <bool has_channels_, bool is_sequence_, typename OutType_, typename InType_,
          int filters_shift_, BoundaryType border_type_ = BoundaryType::REFLECT_101>
struct FilterParams {
  static constexpr int axes = 2;
  static constexpr int filter_dim = axes;
  static constexpr bool has_channels = has_channels_;
  static constexpr bool is_sequence = is_sequence_;
  static constexpr int ndim = static_cast<int>(is_sequence) + axes + static_cast<int>(has_channels);
  static constexpr int filters_shift = filters_shift_;
  static constexpr BoundaryType border_type = border_type_;
  using OutType = OutType_;
  using InType = InType_;
  using WinType = float;
};

template <typename T>
struct FilterGPUTest : public ::testing::Test {
  using InType = typename T::InType;
  using WinType = typename T::WinType;
  using OutType = typename T::OutType;
  using Kernel = Filter2dGpu<OutType, InType, WinType, T::has_channels, T::is_sequence>;

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
    baseline_output_.reshape(GetOutputShape());
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
      int R = filter_shapes_[sample_idx][0];
      int S = filter_shapes_[sample_idx][1];
      int RS = R * S;
      WinType w = 1;
      WinType sum = RS * (RS + 1) / 2;
      for (int x = 0; x < S; x++) {
        for (int y = 0; y < R; y++) {
          filters_view_cpu_[sample_idx].data[y * S + x] = w / sum;
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
    out_view_ = output_.gpu();
    kernel_gpu.Run(ctx_gpu, out_view_, in_view_, filters_view_, anchors_view_, T::border_type);
    FillBaseline();
    auto out_cpu = output_.cpu();

    double eps = std::is_integral<OutType>::value ? 1 : 0.01;
    Check(out_cpu, baseline_out_, EqualEps(eps));
  }

  void FillBaseline() {
    int num_samples = filter_shapes_.num_samples();
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      auto in_view = in_view_cpu_[sample_idx];
      auto out_view = baseline_out_[sample_idx];
      auto filter_view = filters_view_cpu_[sample_idx];
      auto anchor_view = anchors_view_[sample_idx];
      baseline_conv<T::is_sequence, T::has_channels>(in_view, out_view, filter_view, anchor_view);
    }
  }

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

  const TensorListShape<> shape_ch_ = {{29, 145, 128, 3}, {64, 64, 64, 3},   {12, 12, 12, 3},
                                       {16, 512, 512, 1}, {8, 1, 32, 3},     {8, 32, 1, 3},
                                       {1, 8, 32, 3},     {1, 111, 57, 129}, {1, 256, 256, 256},
                                       {16, 1, 517, 3},   {16, 517, 1, 3}};
  const TensorListShape<> shape_noch_ = {{29, 145, 128}, {64, 64, 64}, {12, 12, 12},  {4, 200, 180},
                                         {200, 4, 180},  {75, 75, 75}, {4, 512, 512}, {8, 1, 32},
                                         {8, 32, 1},     {1, 8, 32}};

  const TensorListShape<> filter_shape_base_ = {{3, 3}, {7, 7}, {51, 1}, {1, 51}, {7, 9}, {4, 2}};
  const TensorListShape<> anchors_base = {{1, 1}, {3, 3}, {25, 0}, {0, 25}, {1, 7}, {2, 0}};
};

TYPED_TEST_SUITE_P(FilterGPUTest);

using TestValues = ::testing::Types<
    FilterParams<true, true, float, float, 0>, FilterParams<false, true, float, float, 1>,
    FilterParams<true, false, float, float, 2>, FilterParams<false, false, float, float, 3>,
    FilterParams<true, false, uint8_t, uint8_t, 4>, FilterParams<false, false, uint8_t, uint8_t, 5>,
    FilterParams<true, false, float, uint8_t, 6>, FilterParams<false, false, float, uint8_t, 7>>;

TYPED_TEST_P(FilterGPUTest, DoConvolution) {
  this->RunTest();
}

REGISTER_TYPED_TEST_SUITE_P(FilterGPUTest, DoConvolution);
INSTANTIATE_TYPED_TEST_SUITE_P(Convolution, FilterGPUTest, TestValues);

}  // namespace kernels
}  // namespace dali
