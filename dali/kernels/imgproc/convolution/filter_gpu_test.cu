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

struct SimpleLoader {
  template <typename In>
  void load(In &value, const In *in, int y_idx, int x_idx, int c, int H, int W, int C,
            In fill_value) const {
    (void)fill_value;
    ASSERT_TRUE(y_idx >= 0 && x_idx >= 0 && c >= 0 && y_idx < H && x_idx < W && c < C);
    value = in[y_idx * W * C + x_idx * C + c];
  }
};

template <BoundaryType border_type>
struct BorderHelper {};

template <>
struct BorderHelper<BoundaryType::REFLECT_101> : SimpleLoader {
  int remap(int idx, int len) const {
    return idx_reflect_101(idx, len);
  }
};

template <>
struct BorderHelper<BoundaryType::REFLECT_1001> : SimpleLoader {
  int remap(int idx, int len) const {
    return idx_reflect_1001(idx, len);
  }
};

template <>
struct BorderHelper<BoundaryType::CLAMP> : SimpleLoader {
  int remap(int idx, int len) const {
    return idx_clamp(idx, len);
  }
};

template <>
struct BorderHelper<BoundaryType::WRAP> : SimpleLoader {
  int remap(int idx, int len) const {
    return idx_wrap(idx, len);
  }
};

template <>
struct BorderHelper<BoundaryType::CONSTANT> {
  int remap(int idx, int len) const {
    return idx;
  }

  template <typename In>
  void load(In &value, const In *in, int y_idx, int x_idx, int c, int H, int W, int C,
            In fill_value) const {
    if (y_idx < 0 || y_idx >= H || x_idx < 0 || x_idx >= W) {
      value = fill_value;
    } else {
      value = in[y_idx * W * C + x_idx * C + c];
    }
  }
};

template <>
struct BorderHelper<BoundaryType::ISOLATED> {
  int remap(int idx, int len) const {
    return idx;
  }

  template <typename In>
  void load(In &value, const In *in, int y_idx, int x_idx, int c, int H, int W, int C,
            In fill_value) const {
    ASSERT_TRUE(y_idx >= 0 && x_idx >= 0 && c >= 0 && y_idx < H && x_idx < W && c < C);
    value = in[y_idx * W * C + x_idx * C + c];
  }
};

struct BaselineROI {
  int h_begin, h_end, w_begin, w_end;

  static BaselineROI create(BoundaryType border_type, int H, int W, int R, int S, int anchor_r,
                            int anchor_s) {
    if (border_type != BoundaryType::ISOLATED) {
      return {0, H, 0, W};
    }
    return {-anchor_r, H - (R - 1 + anchor_r), -anchor_s, W - (S - 1 + anchor_s)};
  }
};

template <bool is_sequence, bool has_channels, typename InT, typename OutT, typename WT,
          BoundaryType border_type, int ndim,
          typename Intermediate = decltype(std::declval<InT>() * std::declval<WT>())>
void baseline_conv(const TensorView<StorageCPU, InT, ndim> &in_view,
                   const TensorView<StorageCPU, OutT, ndim> &out_view,
                   const TensorView<StorageCPU, WT, 2> &filter_view,
                   const TensorView<StorageCPU, int, 1> &anchor_view,
                   const BorderHelper<border_type> &border_helper) {
  int F = is_sequence ? in_view.shape[0] : 1;
  int H = in_view.shape[is_sequence];
  int W = in_view.shape[is_sequence + 1];
  int C = has_channels ? in_view.shape[is_sequence + 2] : 1;
  int R = filter_view.shape[0];
  int S = filter_view.shape[1];
  int anchor_r = anchor_view.data[0] == -1 ? -R / 2 : -anchor_view.data[0];
  int anchor_s = anchor_view.data[1] == -1 ? -S / 2 : -anchor_view.data[1];
  auto roi = BaselineROI::create(border_type, H, W, R, S, anchor_r, anchor_s);
  int H_out = out_view.shape[is_sequence];
  int W_out = out_view.shape[is_sequence + 1];
  ASSERT_EQ(H_out, roi.h_end - roi.h_begin);
  ASSERT_EQ(W_out, roi.w_end - roi.w_begin);
  auto *filter_data = filter_view.data;
  for (int f = 0; f < F; f++) {
    const auto *in_data = in_view.data + f * H * W * C;
    auto *out_data = out_view.data + f * H_out * W_out * C;
    for (int y = roi.h_begin; y < roi.h_end; y++) {
      for (int x = roi.w_begin; x < roi.w_end; x++) {
        for (int c = 0; c < C; c++) {
          Intermediate acc = 0;
          for (int r = 0; r < R; r++) {
            int y_idx = border_helper.remap(y + r + anchor_r, H);
            for (int s = 0; s < S; s++) {
              int x_idx = border_helper.remap(x + s + anchor_s, W);
              InT value;
              border_helper.load(value, in_data, y_idx, x_idx, c, H, W, C, static_cast<InT>(0));
              acc += value * filter_data[r * S + s];
            }
          }
          out_data[(y - roi.h_begin) * W_out * C + (x - roi.w_begin) * C + c] =
              ConvertSat<OutT>(acc);
        }
      }
    }
  }
}

template <bool has_channels_, bool is_sequence_, typename OutType_, typename InType_,
          int filters_shift_, BoundaryType border_type_,
          // For isolated border type filters_shift does not apply
          typename Dummy =
              std::enable_if_t<filters_shift_ == 0 || border_type_ != BoundaryType::ISOLATED>>
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

/// @brief Provides initial shapes of inputs for the test casa based on border type.
template <BoundaryType border_type>
struct InputShapes {
  const TensorListShape<> shape_ch = {{29, 145, 128, 3}, {64, 64, 64, 3},   {12, 12, 12, 3},
                                      {16, 512, 512, 1}, {8, 1, 32, 3},     {8, 32, 1, 3},
                                      {1, 8, 32, 3},     {1, 111, 57, 129}, {1, 256, 256, 256},
                                      {16, 1, 517, 3},   {16, 517, 1, 3}};
  const TensorListShape<> shape_noch = {{29, 145, 128}, {64, 64, 64}, {12, 12, 12},  {4, 200, 180},
                                        {200, 4, 180},  {75, 75, 75}, {4, 512, 512}, {8, 1, 32},
                                        {8, 32, 1},     {1, 8, 32}};

  const TensorListShape<> filter_shape_base = {{3, 3}, {7, 7}, {51, 1}, {1, 51}, {7, 9}, {4, 2}};
  const TensorListShape<> anchors_base = {{1, 1}, {3, 3}, {25, 0}, {0, 25}, {1, 7}, {2, 0}};
};


/// @brief Usually it is desireble to include cases when the filter window size exceeds
/// the input sample shape. However in istolated border mode it is forbidden and would result
/// in validation error.
template <>
struct InputShapes<BoundaryType::ISOLATED> {
  const TensorListShape<> shape_ch = {{29, 145, 128, 3}, {64, 64, 64, 3},    {12, 12, 12, 3},
                                      {16, 512, 512, 1}, {8, 2, 32, 3},      {8, 32, 2, 3},
                                      {1, 111, 57, 129}, {1, 256, 256, 256}, {1, 255, 255, 255},
                                      {16, 1, 517, 3},   {16, 517, 1, 3}};
  const TensorListShape<> shape_noch = {{29, 146, 127}, {64, 63, 65}, {12, 12, 12},  {4, 200, 180},
                                        {50, 14, 180},  {75, 75, 75}, {4, 512, 512}, {8, 256, 256},
                                        {8, 255, 255},  {8, 1, 32},   {8, 32, 1}};

  const TensorListShape<> filter_shape_base = {{3, 3},     {7, 7}, {11, 11}, {7, 9},
                                               {1, 2},     {2, 2}, {4, 4},   {255, 255},
                                               {255, 255}, {1, 3}, {1, 1}};
  const TensorListShape<> anchors_base = {{1, 1},  {3, 3},  {4, 4}, {1, 7},   {0, 0},  {1, 1},
                                          {-1, 1}, {1, -1}, {3, 2}, {-1, -1}, {-1, -1}};
};

template <typename T>
struct FilterGPUTest : public ::testing::Test {
  using InType = typename T::InType;
  using WinType = typename T::WinType;
  using OutType = typename T::OutType;
  using Kernel = Filter2dGpu<OutType, InType, WinType, T::has_channels, T::is_sequence>;

  TensorListShape<T::ndim> GetInputShape() {
    if (T::has_channels) {
      return input_shapes_.shape_ch.template last<T::ndim>();
    } else {
      return input_shapes_.shape_noch.template last<T::ndim>();
    }
  }

  void FillAnchors(int num_samples) {
    anchors_.reshape(uniform_list_shape<1>(num_samples, {T::filter_dim}));
    anchors_view_ = anchors_.cpu();
    int num_base_anchors = input_shapes_.anchors_base.num_samples();
    ASSERT_TRUE(num_base_anchors == input_shapes_.filter_shape_base.num_samples());
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      int idx = (sample_idx + T::filters_shift) % num_base_anchors;
      anchors_view_[sample_idx].data[0] = input_shapes_.anchors_base[idx][0];
      anchors_view_[sample_idx].data[1] = input_shapes_.anchors_base[idx][1];
    }
  }

  TensorListShape<T::filter_dim> GetFilterShape(int num_samples) {
    TensorListShape<T::filter_dim> filter_shapes(num_samples);
    int num_base_filters = input_shapes_.filter_shape_base.num_samples();
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      int idx = (sample_idx + T::filters_shift) % num_base_filters;
      TensorShape<2> shape = input_shapes_.filter_shape_base[idx];
      filter_shapes.set_tensor_shape(sample_idx, shape);
    }
    return filter_shapes;
  }

  TensorListShape<T::ndim> GetOutputShape(TensorListShape<T::ndim> shapes,
                                          const TensorListShape<T::filter_dim> &filter_shapes) {
    if (T::border_type == BoundaryType::ISOLATED) {
      for (int sample_idx = 0; sample_idx < shapes.num_samples(); sample_idx++) {
        auto shape = shapes[sample_idx];
        const auto &filter_shape = filter_shapes[sample_idx];
        for (int dim_idx = 0; dim_idx < T::filter_dim; dim_idx++) {
          shape[T::is_sequence + dim_idx] -= filter_shape[dim_idx] - 1;
          if (shape[T::is_sequence + dim_idx] <= 0) {
            throw std::logic_error("incorrect test shapes for isolated border mode");
          }
        }
        shapes.set_tensor_shape(sample_idx, shape);
      }
    }
    return shapes;
  }

  void FillFilters(const TensorListShape<T::filter_dim> &filter_shapes) {
    filters_.reshape(filter_shapes);
    filters_view_cpu_ = filters_.cpu();
    for (int sample_idx = 0; sample_idx < filter_shapes.num_samples(); sample_idx++) {
      int R = filter_shapes[sample_idx][0];
      int S = filter_shapes[sample_idx][1];
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

  void SetUp() override {
    auto in_shapes = GetInputShape();
    int num_samples = in_shapes.num_samples();
    input_.reshape(in_shapes);
    in_view_cpu_ = input_.cpu();
    std::mt19937 rng;
    UniformRandomFill(in_view_cpu_, rng, 0, 64);
    in_view_ = input_.gpu();
    auto filter_shapes = GetFilterShape(num_samples);
    FillAnchors(num_samples);
    FillFilters(filter_shapes);
    auto output_shape = GetOutputShape(in_shapes, filter_shapes);
    output_.reshape(output_shape);
    baseline_output_.reshape(output_shape);
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
    FillBaseline(BorderHelper<T::border_type>{});
    auto out_cpu = output_.cpu();

    if (std::is_integral<OutType>::value) {
      Check(out_cpu, baseline_out_, EqualEps(1));
    } else {
      Check(out_cpu, baseline_out_, EqualEpsRel());
    }
  }

  template <typename Border>
  void FillBaseline(Border &&border_helper) {
    int num_samples = in_view_.num_samples();
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      auto in_view = in_view_cpu_[sample_idx];
      auto out_view = baseline_out_[sample_idx];
      auto filter_view = filters_view_cpu_[sample_idx];
      auto anchor_view = anchors_view_[sample_idx];
      baseline_conv<T::is_sequence, T::has_channels>(in_view, out_view, filter_view, anchor_view,
                                                     border_helper);
    }
  }

  InputShapes<T::border_type> input_shapes_;

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
};

TYPED_TEST_SUITE_P(FilterGPUTest);

using TestValues =
    ::testing::Types<FilterParams<true, true, float, float, 0, BoundaryType::REFLECT_101>,
                     FilterParams<false, true, float, float, 1, BoundaryType::REFLECT_1001>,
                     FilterParams<true, false, float, float, 2, BoundaryType::CLAMP>,
                     FilterParams<false, false, float, float, 3, BoundaryType::WRAP>,
                     FilterParams<true, false, uint8_t, uint8_t, 4, BoundaryType::CONSTANT>,
                     FilterParams<false, false, uint8_t, uint8_t, 0, BoundaryType::ISOLATED>,
                     FilterParams<true, false, uint8_t, uint8_t, 0, BoundaryType::ISOLATED>,
                     FilterParams<true, false, float, uint8_t, 5, BoundaryType::REFLECT_101>,
                     FilterParams<false, false, float, uint8_t, 6, BoundaryType::REFLECT_1001>,
                     FilterParams<true, true, int32_t, int32_t, 7, BoundaryType::CLAMP>,
                     FilterParams<false, true, int32_t, int32_t, 8, BoundaryType::WRAP>,
                     FilterParams<true, true, float, int32_t, 9, BoundaryType::CONSTANT>,
                     FilterParams<false, true, float, int32_t, 0, BoundaryType::ISOLATED>,
                     FilterParams<true, true, float, int32_t, 0, BoundaryType::ISOLATED>>;

TYPED_TEST_P(FilterGPUTest, DoConvolution) {
  this->RunTest();
}

REGISTER_TYPED_TEST_SUITE_P(FilterGPUTest, DoConvolution);
INSTANTIATE_TYPED_TEST_SUITE_P(Convolution, FilterGPUTest, TestValues);

}  // namespace kernels
}  // namespace dali
