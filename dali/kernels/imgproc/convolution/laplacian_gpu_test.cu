// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cmath>
#include <vector>

#include "dali/kernels/common/utils.h"
#include "dali/kernels/imgproc/convolution/laplacian_cpu.h"
#include "dali/kernels/imgproc/convolution/laplacian_gpu.cuh"
#include "dali/kernels/imgproc/convolution/laplacian_test.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace kernels {

template <typename Out_, typename In_, int axes_, bool has_channels_, bool is_sequence_,
          bool use_smoothing_>
struct test_laplacian {
  static constexpr int axes = axes_;
  static constexpr bool has_channels = has_channels_;
  static constexpr bool is_sequence = is_sequence_;
  static constexpr bool use_smoothing = use_smoothing_;
  using Out = Out_;
  using In = In_;
};

/**
 * @brief Compares GPU implementation against CPU implementation.
 */
template <typename T>
struct LaplacianGpuTest : public ::testing::Test {
  static constexpr int max_window_size = 23;
  static constexpr int max_sample_dim = 5;
  static constexpr int max_axes = 3;
  static constexpr bool has_channels = T::has_channels;
  static constexpr bool is_sequence = T::is_sequence;
  static constexpr bool use_smoothing = T::use_smoothing;
  static constexpr int axes = T::axes;
  static constexpr int sample_ndim = axes + static_cast<int>(has_channels);
  static constexpr int ndim = sample_ndim + static_cast<int>(is_sequence);
  using Out = typename T::Out;
  using In = typename T::In;
  using W = float;
  using KernelCpu = LaplacianCpu<Out, In, W, axes, has_channels>;
  using KernelGpu = LaplacianGpu<Out, In, W, axes, has_channels, is_sequence>;

  static TensorListShape<ndim> GetShape() {
    static const TensorListShape<> shapes = {
        {7, 29, 145, 128}, {3, 64, 64, 64},  {4, 164, 164, 164}, {11, 12, 12, 12},
        {4, 4, 200, 180},  {1, 200, 4, 180}, {1, 75, 75, 75},    {2, 16, 256, 256}};
    static const TensorListShape<> channels = {{3}, {3}, {1}, {5}, {7}, {3}, {5}, {1}};
    if (!has_channels) {
      return shapes.template first<ndim>();
    } else {
      auto shape = shapes.template first<ndim - 1>();
      TensorListShape<ndim> result(shapes.num_samples());
      for (int i = 0; i < shapes.num_samples(); i++) {
        result.set_tensor_shape(i, shape_cat(shape[i], channels[i]));
      }
      return result;
    }
  }

  static TensorListShape<axes> GetWindowSize() {
    static const TensorListShape<> window_sizes = {{3, 5, 7}, {7, 5, 3},   {5, 5, 5},
                                                   {3, 3, 3}, {11, 11, 5}, {13, 15, 13},
                                                   {7, 9, 7}, {23, 19, 17}};
    return window_sizes.template last<axes>();
  }

  static TensorListShape<axes> GetSmoothingSize() {
    // use 1 as the middle window size in a whole batch to test if optimization that removes
    // unnecessary smoothing convolutions on per partial derivative basis gives correct results
    static const TensorListShape<> window_sizes = {{3, 1, 1}, {5, 1, 9},   {7, 1, 7}, {1, 1, 7},
                                                   {7, 1, 5}, {13, 1, 13}, {7, 1, 9}, {23, 1, 17}};
    return window_sizes.template last<axes>();
  }

  void FillWindows() {
    // get per sample x per axis window sizes
    auto deriv_sizes = GetWindowSize();
    int nsamples = deriv_sizes.num_samples();
    auto smoothing_sizes =
        use_smoothing ? GetSmoothingSize() : uniform_list_shape(nsamples, uniform_array<axes>(1));
    // flatten window sizes
    TensorListShape<1> flat_deriv_sizes;
    TensorListShape<1> flat_smoothing_sizes;
    flat_deriv_sizes.resize(nsamples * axes);
    flat_smoothing_sizes.resize(nsamples * axes);
    for (int i = 0; i < nsamples; i++) {
      for (int axis = 0; axis < axes; axis++) {
        flat_deriv_sizes.set_tensor_shape(i * axes + axis, {deriv_sizes[i][axis]});
        flat_smoothing_sizes.set_tensor_shape(i * axes + axis, {smoothing_sizes[i][axis]});
      }
    }
    deriv_windows_.reshape(flat_deriv_sizes);
    smoothing_windows_.reshape(flat_smoothing_sizes);
    deriv_win_ = deriv_windows_.cpu();
    smoothing_win_ = smoothing_windows_.cpu();
    for (int i = 0; i < deriv_win_.num_samples(); i++) {
      FillSobelWindow(make_span(deriv_win_[i].data, deriv_win_[i].num_elements()), 2);
    }
    for (int i = 0; i < smoothing_win_.num_samples(); i++) {
      FillSobelWindow(make_span(smoothing_win_[i].data, smoothing_win_[i].num_elements()), 0);
    }
  }

  void SetUp() override {
    FillWindows();

    auto shapes = GetShape();
    input_.reshape(shapes);
    baseline_in_ = input_.cpu();

    std::mt19937 rng;
    UniformRandomFill(baseline_in_, rng, 0, 64);
    in_ = input_.gpu();

    output_.reshape(shapes);
    baseline_output_.reshape(shapes);

    int nsamples = shapes.size();
    for (int i = 0; i < axes; i++) {
      for (int j = 0; j < axes; j++) {
        win_sizes_[i][j].resize(nsamples);
        windows_[i][j].resize(nsamples);
      }
      scales_[i].resize(nsamples);
      scale_spans_[i] = make_span(scales_[i]);
    }
    for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
      for (int i = 0; i < axes; i++) {
        int win_size_sum = -axes - 2;
        for (int j = 0; j < axes; j++) {
          auto& windows = i == j ? deriv_win_ : smoothing_win_;
          auto window = windows[sample_idx * axes + j];
          win_size_sum += window.shape.num_elements();
          win_sizes_[i][j].set_tensor_shape(sample_idx, window.shape);
          windows_[i][j].data[sample_idx] = window.data;
          windows_[i][j].shape.set_tensor_shape(sample_idx, window.shape);
        }
        scales_[i][sample_idx] = std::exp2f(-win_size_sum);
      }
    }
  }

  void RunTest() {
    KernelContext ctx_cpu = {}, ctx_gpu = {};
    ctx_gpu.gpu.stream = 0;
    KernelCpu kernel_cpu;
    KernelGpu kernel_gpu;
    int nsamples = in_.shape.size();
    baseline_out_ = baseline_output_.cpu();
    out_ = output_.gpu();

    std::array<bool, axes> has_smoothing = uniform_array<axes>(false);
    for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
      std::array<std::array<int, axes>, axes> window_size;
      std::array<std::array<TensorView<StorageCPU, const W, 1>, axes>, axes> windows;
      std::array<float, axes> scales;
      for (int i = 0; i < axes; i++) {
        for (int j = 0; j < axes; j++) {
          if (i != j && win_sizes_[i][j][sample_idx].num_elements() > 1) {
            has_smoothing[i] = true;
          }
          window_size[i][j] = win_sizes_[i][j][sample_idx].num_elements();
          windows[i][j] = windows_[i][j][sample_idx];
        }
        scales[i] = scales_[i][sample_idx];
      }
      auto elem_shape = baseline_in_.shape[sample_idx].template last<sample_ndim>();
      auto req = kernel_cpu.Setup(ctx_cpu, elem_shape, window_size);

      const auto& shape = baseline_in_.shape[sample_idx];
      auto elem_volume = volume(shape.begin() + static_cast<int>(is_sequence), shape.end());
      int seq_elements = volume(shape.begin(), shape.begin() + static_cast<int>(is_sequence));
      int64_t stride = elem_volume;

      for (int elem_idx = 0; elem_idx < seq_elements; elem_idx++) {
        auto in_view = TensorView<StorageCPU, const In, sample_ndim>{
            baseline_in_[sample_idx].data + stride * elem_idx, elem_shape};
        auto out_view = TensorView<StorageCPU, Out, sample_ndim>{
            baseline_out_[sample_idx].data + stride * elem_idx, elem_shape};
        // Copy context so that the kernel instance can modify scratchpad
        DynamicScratchpad scratchpad;
        ctx_cpu.scratchpad = &scratchpad;
        kernel_cpu.Run(ctx_cpu, out_view, in_view, windows, scales);
      }
    }

    for (int i = 0; i < axes; i++) {
      if (!has_smoothing[i]) {
        for (int j = 0; j < axes; j++) {
          if (i != j) {
            win_sizes_[i][j].resize(0);
            windows_[i][j].resize(0);
          }
        }
      }
    }

    auto req = kernel_gpu.Setup(ctx_gpu, in_.shape, win_sizes_);

    DynamicScratchpad scratchpad;
    ctx_gpu.scratchpad = &scratchpad;
    kernel_gpu.Run(ctx_gpu, out_, in_, windows_, scale_spans_);

    auto out_cpu_ = output_.cpu();

    double eps = std::is_integral<Out>::value ? 1 : 0.01;
    Check(out_cpu_, baseline_out_, EqualEps(eps));
  }

  TestTensorList<W, 1> deriv_windows_;
  TestTensorList<W, 1> smoothing_windows_;
  TestTensorList<In, ndim> input_;
  TestTensorList<Out, ndim> output_;
  TestTensorList<Out, ndim> baseline_output_;

  TensorListView<StorageCPU, W, 1> deriv_win_;
  TensorListView<StorageCPU, W, 1> smoothing_win_;
  TensorListView<StorageGPU, In, ndim> in_;
  TensorListView<StorageGPU, Out, ndim> out_;
  TensorListView<StorageCPU, In, ndim> baseline_in_;
  TensorListView<StorageCPU, Out, ndim> baseline_out_;

  std::array<std::array<TensorListShape<1>, axes>, axes> win_sizes_;
  std::array<std::array<TensorListView<StorageCPU, const float, 1>, axes>, axes> windows_;
  std::array<std::vector<float>, axes> scales_;
  std::array<span<const float>, axes> scale_spans_;
};

TYPED_TEST_SUITE_P(LaplacianGpuTest);

using LaplacianTestValues =
    ::testing::Types<test_laplacian<float, float, 1, true, true, false>,
                     test_laplacian<float, float, 1, true, false, false>,
                     test_laplacian<float, float, 1, false, true, false>,
                     test_laplacian<float, float, 1, false, false, false>,
                     test_laplacian<float, float, 2, true, true, false>,
                     test_laplacian<float, float, 2, true, false, false>,
                     test_laplacian<float, float, 2, false, true, false>,
                     test_laplacian<float, float, 2, false, false, false>,
                     test_laplacian<float, float, 2, true, true, true>,
                     test_laplacian<float, float, 2, true, false, true>,
                     test_laplacian<float, float, 2, false, true, true>,
                     test_laplacian<float, float, 2, false, false, true>,
                     test_laplacian<float, float, 3, true, true, false>,
                     test_laplacian<float, float, 3, true, false, false>,
                     test_laplacian<float, float, 3, false, true, false>,
                     test_laplacian<float, float, 3, false, false, false>,
                     test_laplacian<float, float, 3, true, true, true>,
                     test_laplacian<float, float, 3, true, false, true>,
                     test_laplacian<float, float, 3, false, true, true>,
                     test_laplacian<float, float, 3, false, false, true>,

                     test_laplacian<uint8_t, uint8_t, 1, true, true, true>,
                     test_laplacian<uint8_t, uint8_t, 2, true, true, true>,
                     test_laplacian<uint8_t, uint8_t, 3, true, true, true>,
                     test_laplacian<uint8_t, uint8_t, 1, true, true, false>,
                     test_laplacian<uint8_t, uint8_t, 2, true, true, false>,
                     test_laplacian<uint8_t, uint8_t, 3, true, true, false>>;

TYPED_TEST_P(LaplacianGpuTest, DoLaplacian) {
  this->RunTest();
}

REGISTER_TYPED_TEST_SUITE_P(LaplacianGpuTest, DoLaplacian);
INSTANTIATE_TYPED_TEST_SUITE_P(LaplacianGpuKernel, LaplacianGpuTest, LaplacianTestValues);

}  // namespace kernels
}  // namespace dali
