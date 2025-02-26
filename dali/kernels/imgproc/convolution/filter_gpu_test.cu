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
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace kernels {

using namespace boundary;  // NOLINT(build/namespaces)

struct SimpleLoader {
  template <typename In>
  void load(In &value, const In *in, int z_idx, int y_idx, int x_idx, int c, int D, int H, int W,
            int C, In fill_value) const {
    (void)fill_value;
    ASSERT_TRUE(z_idx >= 0 && y_idx >= 0 && x_idx >= 0 && c >= 0 && z_idx < D && y_idx < H &&
                x_idx < W && c < C);
    value = in[z_idx * H * W * C + y_idx * W * C + x_idx * C + c];
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
  void load(In &value, const In *in, int z_idx, int y_idx, int x_idx, int c, int D, int H, int W,
            int C, In fill_value) const {
    if (z_idx < 0 || z_idx >= D || y_idx < 0 || y_idx >= H || x_idx < 0 || x_idx >= W) {
      value = fill_value;
    } else {
      value = in[z_idx * H * W * C + y_idx * W * C + x_idx * C + c];
    }
  }
};

struct BorderHelperAssertValid {
  int remap(int idx, int len) const {
    return idx;
  }

  template <typename In>
  void load(In &value, const In *in, int z_idx, int y_idx, int x_idx, int c, int D, int H, int W,
            int C, In fill_value) const {
    ASSERT_TRUE(z_idx >= 0 && y_idx >= 0 && x_idx >= 0 && c >= 0 && z_idx < D && y_idx < H &&
                x_idx < W && c < C);
    value = in[z_idx * H * W * C + y_idx * W * C + x_idx * C + c];
  }
};

void anchor_prs(int &p, int &r, int &s, const ivec2 &anchor) {
  p = 0;
  r = anchor.y;
  s = anchor.x;
}

void anchor_prs(int &p, int &r, int &s, const ivec3 &anchor) {
  p = anchor.z;
  r = anchor.y;
  s = anchor.x;
}

template <bool is_sequence, bool has_channels, typename InT, typename OutT, typename WT, int ndim,
          int axes, typename Border,
          typename Intermediate = decltype(std::declval<InT>() * std::declval<WT>())>
void baseline_conv(const TensorView<StorageCPU, InT, ndim> &in_view,
                   const TensorView<StorageCPU, OutT, ndim> &out_view,
                   const TensorView<StorageCPU, WT, axes> &filter_view, const ivec<axes> anchor,
                   const Border &border_helper) {
  static constexpr bool is_vol = axes == 3;
  int F = is_sequence ? in_view.shape[0] : 1;
  int D = is_vol ? in_view.shape[is_sequence] : 1;
  int H = in_view.shape[is_vol + is_sequence];
  int W = in_view.shape[is_vol + is_sequence + 1];
  int C = has_channels ? in_view.shape[is_vol + is_sequence + 2] : 1;
  int P = is_vol ? filter_view.shape[0] : 1;
  int R = filter_view.shape[is_vol];
  int S = filter_view.shape[is_vol + 1];
  int D_out = is_vol ? out_view.shape[is_sequence] : 1;
  int H_out = out_view.shape[is_vol + is_sequence];
  int W_out = out_view.shape[is_vol + is_sequence + 1];
  int anchor_p, anchor_r, anchor_s;
  anchor_prs(anchor_p, anchor_r, anchor_s, anchor);
  auto *filter_data = filter_view.data;
  for (int f = 0; f < F; f++) {
    const auto *in_data = in_view.data + f * D * H * W * C;
    auto *out_data = out_view.data + f * D_out * H_out * W_out * C;
    for (int z = 0; z < D_out; z++) {
      for (int y = 0; y < H_out; y++) {
        for (int x = 0; x < W_out; x++) {
          for (int c = 0; c < C; c++) {
            Intermediate acc = 0;
            for (int p = 0; p < P; p++) {
              int z_idx = border_helper.remap(z + p - anchor_p, D);
              for (int r = 0; r < R; r++) {
                int y_idx = border_helper.remap(y + r - anchor_r, H);
                for (int s = 0; s < S; s++) {
                  int x_idx = border_helper.remap(x + s - anchor_s, W);
                  InT value;
                  border_helper.load(value, in_data, z_idx, y_idx, x_idx, c, D, H, W, C,
                                     static_cast<InT>(0));
                  acc += value * filter_data[p * R * S + r * S + s];
                }
              }
            }
            out_data[z * H_out * W_out * C + y * W_out * C + x * C + c] = ConvertSat<OutT>(acc);
          }
        }
      }
    }
  }
}

template <int axes_, bool has_channels_, bool is_sequence_, typename OutType_, typename InType_,
          int filters_shift_, BoundaryType border_type_, bool valid_only_mode_,
          // For isolated border type filters_shift does not apply
          typename Dummy = std::enable_if_t<!valid_only_mode_ || filters_shift_ == 0>>
struct FilterParams {
  static constexpr int axes = axes_;
  static constexpr bool has_channels = has_channels_;
  static constexpr bool is_sequence = is_sequence_;
  static constexpr int ndim = static_cast<int>(is_sequence) + axes + static_cast<int>(has_channels);
  static constexpr int filters_shift = filters_shift_;
  static constexpr BoundaryType border_type = border_type_;
  static constexpr bool valid_only_mode = valid_only_mode_;
  using OutType = OutType_;
  using InType = InType_;
  using WinType = float;
  static_assert(axes == 2 || axes == 3);
};

/// @brief Provides initial shapes of inputs for the test casa based on border type.
template <bool valid_only_mode>
struct InputShapes {
  const TensorListShape<> shape_ch = {
      {1, 29, 145, 128, 3}, {2, 64, 64, 64, 3}, {13, 12, 12, 12, 3}, {3, 16, 512, 512, 1},
      {7, 8, 1, 32, 3},     {4, 8, 32, 1, 3},   {128, 1, 8, 32, 3},  {7, 1, 111, 57, 129},
      {1, 1, 64, 4, 256},   {4, 16, 1, 517, 3}, {7, 16, 517, 1, 3},  {3, 517, 16, 1, 3}};
  const TensorListShape<> shape_noch = {
      {2, 29, 145, 128}, {1, 64, 64, 64},  {7, 12, 12, 12}, {1, 4, 200, 180}, {5, 200, 4, 180},
      {1, 75, 75, 75},   {1, 4, 512, 512}, {3, 8, 1, 32},   {2, 8, 32, 1},    {128, 1, 8, 32}};

  const TensorListShape<> filter_shape_base = {{3, 3, 3},  {7, 7, 7}, {51, 1, 1}, {1, 51, 1},
                                               {1, 1, 51}, {3, 7, 9}, {8, 4, 2}};
  const TensorListShape<> anchors_base = {{1, 1, 1},  {3, 3, 3}, {25, 0, 0}, {0, 25, 0},
                                          {0, 0, 25}, {2, 1, 7}, {6, 2, 0}};
};


/// @brief Usually it is desireble to include cases when the filter window size exceeds
/// the input sample shape. However in valid only mode it is forbidden and would result
/// in validation error.
template <>
struct InputShapes<true> {
  const TensorListShape<> shape_ch = {{1, 29, 145, 128, 3}, {2, 64, 64, 64, 3}, {3, 12, 12, 12, 3},
                                      {1, 16, 512, 512, 1}, {13, 8, 2, 32, 3},  {2, 8, 32, 2, 3},
                                      {3, 4, 111, 57, 129}, {1, 1, 256, 4, 16}, {1, 256, 1, 4, 16},
                                      {2, 1, 16, 4, 255},   {1, 16, 1, 517, 3}, {1, 16, 517, 1, 3}};
  const TensorListShape<> shape_noch = {{1, 29, 146, 127}, {2, 64, 63, 65},  {41, 12, 12, 12},
                                        {1, 4, 200, 180},  {2, 50, 14, 180}, {1, 75, 75, 75},
                                        {2, 4, 512, 512},  {1, 1, 256, 4},   {1, 256, 1, 4},
                                        {2, 1, 16, 4},     {1, 8, 1, 32},    {3, 8, 32, 1}};

  const TensorListShape<> filter_shape_base = {{3, 3, 3},   {7, 7, 7}, {11, 11, 11}, {4, 7, 9},
                                               {1, 1, 2},   {2, 2, 2}, {4, 4, 4},    {1, 256, 4},
                                               {255, 1, 4}, {1, 5, 4}, {1, 1, 3},    {1, 1, 1}};
  const TensorListShape<> anchors_base = {{1, 1, 1},    {3, 3, 3},    {4, 4, 4},    {0, 1, 7},
                                          {0, 0, 0},    {1, 1, 1},    {-1, -1, 1},  {1, 1, -1},
                                          {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}};
};

template <typename T>
struct FilterGPUTest : public ::testing::Test {
  using InType = typename T::InType;
  using WinType = typename T::WinType;
  using OutType = typename T::OutType;
  using Kernel = FilterGpu<OutType, InType, WinType, T::has_channels, T::is_sequence, T::axes,
                           T::valid_only_mode>;

  TensorListShape<T::ndim> GetInputShape() {
    if (T::has_channels) {
      return input_shapes_.shape_ch.template last<T::ndim>();
    } else {
      return input_shapes_.shape_noch.template last<T::ndim>();
    }
  }

  void FillAnchors(const TensorListShape<T::axes> &filter_shapes) {
    int num_samples = filter_shapes.num_samples();
    anchors_.resize(num_samples);
    auto anchors_base = input_shapes_.anchors_base.template last<T::axes>();
    int num_base_anchors = anchors_base.num_samples();
    ASSERT_TRUE(num_base_anchors == input_shapes_.filter_shape_base.num_samples());
    if (T::valid_only_mode) {
      for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
        anchors_[sample_idx] = 0;
      }
    } else {
      for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
        int idx = (sample_idx + T::filters_shift) % num_base_anchors;
        for (int dim = 0; dim < T::axes; dim++) {
          auto anchor = anchors_base[idx][dim];
          if (anchor == -1) {
            anchor = filter_shapes[sample_idx][dim] / 2;
          }
          anchors_[sample_idx][T::axes - 1 - dim] = anchor;
        }
      }
    }
  }

  TensorListShape<T::axes> GetFilterShape(int num_samples) {
    TensorListShape<T::axes> filter_shapes(num_samples);
    auto filter_shape_base = input_shapes_.filter_shape_base.template last<T::axes>();
    int num_base_filters = filter_shape_base.num_samples();
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      int idx = (sample_idx + T::filters_shift) % num_base_filters;
      auto shape = filter_shape_base[idx];
      filter_shapes.set_tensor_shape(sample_idx, shape);
    }
    return filter_shapes;
  }

  TensorListShape<T::ndim> GetOutputShape(TensorListShape<T::ndim> shapes,
                                          const TensorListShape<T::axes> &filter_shapes) {
    if (T::valid_only_mode) {
      for (int sample_idx = 0; sample_idx < shapes.num_samples(); sample_idx++) {
        auto shape = shapes[sample_idx];
        const auto &filter_shape = filter_shapes[sample_idx];
        for (int dim_idx = 0; dim_idx < T::axes; dim_idx++) {
          shape[T::is_sequence + dim_idx] -= filter_shape[dim_idx] - 1;
          if (shape[T::is_sequence + dim_idx] <= 0) {
            throw std::logic_error("incorrect test shapes for valid only mode for sample");
          }
        }
        shapes.set_tensor_shape(sample_idx, shape);
      }
    }
    return shapes;
  }

  void FillFilters(const TensorListShape<T::axes> &filter_shapes) {
    static constexpr bool is_vol = T::axes == 3;
    filters_.reshape(filter_shapes);
    filters_view_cpu_ = filters_.cpu();
    for (int sample_idx = 0; sample_idx < filter_shapes.num_samples(); sample_idx++) {
      int P = is_vol ? filter_shapes[sample_idx][0] : 1;
      int R = filter_shapes[sample_idx][is_vol];
      int S = filter_shapes[sample_idx][is_vol + 1];
      int PRS = P * R * S;
      WinType w = 1;
      WinType sum = PRS * (PRS + 1) / 2;
      for (int z = 0; z < P; z++) {
        for (int y = 0; y < R; y++) {
          for (int x = 0; x < S; x++) {
            filters_view_cpu_[sample_idx].data[z * R * S + y * S + x] = w / sum;
            w += 1;
          }
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
    FillFilters(filter_shapes);
    FillAnchors(filter_shapes);
    auto output_shape = GetOutputShape(in_shapes, filter_shapes);
    output_.reshape(output_shape);
    baseline_output_.reshape(output_shape);
  }

  void RunTest() {
    KernelContext ctx_gpu;
    ctx_gpu.gpu.stream = 0;
    DynamicScratchpad dyn_scratchpad(AccessOrder(ctx_gpu.gpu.stream));
    ctx_gpu.scratchpad = &dyn_scratchpad;
    Kernel kernel_gpu;

    auto data_shape = GetInputShape();
    int num_samples = data_shape.size();

    baseline_out_ = baseline_output_.cpu();
    out_view_ = output_.gpu();
    kernel_gpu.Run(ctx_gpu, out_view_, in_view_, filters_view_, make_cspan(anchors_),
                   T::border_type);
    if (!T::valid_only_mode) {
      FillBaseline(BorderHelper<T::border_type>{});
    } else {
      FillBaseline(BorderHelperAssertValid{});
    }
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
      baseline_conv<T::is_sequence, T::has_channels>(in_view, out_view, filter_view,
                                                     anchors_[sample_idx], border_helper);
    }
  }

  InputShapes<T::valid_only_mode> input_shapes_;

  TestTensorList<WinType, T::axes> filters_;
  TestTensorList<InType, T::ndim> input_;
  TestTensorList<OutType, T::ndim> output_;
  TestTensorList<OutType, T::ndim> baseline_output_;

  std::vector<ivec<T::axes>> anchors_;

  TensorListView<StorageCPU, WinType, T::axes> filters_view_cpu_;
  TensorListView<StorageGPU, WinType, T::axes> filters_view_;
  TensorListView<StorageCPU, InType, T::ndim> in_view_cpu_;
  TensorListView<StorageGPU, InType, T::ndim> in_view_;
  TensorListView<StorageGPU, OutType, T::ndim> out_view_;
  TensorListView<StorageCPU, OutType, T::ndim> baseline_out_;
};

TYPED_TEST_SUITE_P(FilterGPUTest);

using TestValues = ::testing::Types<
    FilterParams<3, true, true, float, float, 0, BoundaryType::REFLECT_101, false>,
    FilterParams<3, false, true, float, float, 1, BoundaryType::REFLECT_1001, false>,
    FilterParams<3, true, false, float, float, 2, BoundaryType::CLAMP, false>,
    FilterParams<3, false, false, float, float, 3, BoundaryType::WRAP, false>,
    FilterParams<3, true, false, uint8_t, uint8_t, 4, BoundaryType::CONSTANT, false>,
    FilterParams<3, false, false, uint8_t, uint8_t, 0, BoundaryType::REFLECT_101, true>,
    FilterParams<3, true, false, uint8_t, uint8_t, 0, BoundaryType::REFLECT_1001, true>,
    FilterParams<3, true, false, float, uint8_t, 5, BoundaryType::REFLECT_101, false>,
    FilterParams<3, false, false, float, uint8_t, 6, BoundaryType::REFLECT_1001, false>,
    FilterParams<3, true, true, int32_t, int32_t, 7, BoundaryType::CLAMP, false>,
    FilterParams<3, false, true, int32_t, int32_t, 8, BoundaryType::WRAP, false>,
    FilterParams<3, true, true, float, int32_t, 9, BoundaryType::CONSTANT, false>,
    FilterParams<3, false, true, float, int32_t, 0, BoundaryType::CLAMP, true>,
    FilterParams<3, true, true, float, int32_t, 0, BoundaryType::CONSTANT, true>,

    FilterParams<2, true, true, float, float, 0, BoundaryType::REFLECT_101, false>,
    FilterParams<2, false, true, float, float, 1, BoundaryType::REFLECT_1001, false>,
    FilterParams<2, true, false, float, float, 2, BoundaryType::CLAMP, false>,
    FilterParams<2, false, false, float, float, 3, BoundaryType::WRAP, false>,
    FilterParams<2, true, false, uint8_t, uint8_t, 4, BoundaryType::CONSTANT, false>,
    FilterParams<2, false, false, uint8_t, uint8_t, 0, BoundaryType::REFLECT_101, true>,
    FilterParams<2, true, false, uint8_t, uint8_t, 0, BoundaryType::REFLECT_1001, true>,
    FilterParams<2, true, false, float, uint8_t, 5, BoundaryType::REFLECT_101, false>,
    FilterParams<2, false, false, float, uint8_t, 6, BoundaryType::REFLECT_1001, false>,
    FilterParams<2, true, true, int32_t, int32_t, 7, BoundaryType::CLAMP, false>,
    FilterParams<2, false, true, int32_t, int32_t, 8, BoundaryType::WRAP, false>,
    FilterParams<2, true, true, float, int32_t, 9, BoundaryType::CONSTANT, false>,
    FilterParams<2, false, true, float, int32_t, 0, BoundaryType::CLAMP, true>,
    FilterParams<2, true, true, float, int32_t, 0, BoundaryType::CONSTANT, true>>;

TYPED_TEST_P(FilterGPUTest, ApplyConvolution) {
  this->RunTest();
}

REGISTER_TYPED_TEST_SUITE_P(FilterGPUTest, ApplyConvolution);
INSTANTIATE_TYPED_TEST_SUITE_P(Convolution, FilterGPUTest, TestValues);

}  // namespace kernels
}  // namespace dali
