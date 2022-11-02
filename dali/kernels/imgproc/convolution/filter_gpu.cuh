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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_FILTER_GPU_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_FILTER_GPU_H_

#include <limits>
#include <utility>
#include <vector>

#include "dali/core/boundary.h"
#include "dali/core/convert.h"
#include "dali/core/cuda_rt_utils.h"
#include "dali/core/static_switch.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

namespace filter {

struct ShapeDesc {
  int64_t hwc;
  int wc, f, h, w, c;
  int filter_vol, r, s;
  int filter_top_anchor, filter_left_anchor;
  int in_workspace_width;
};

template <typename Out_, typename In_, typename W_, typename Acc_, int lanes_>
struct SampleDesc {
  using Acc = Acc_;
  using Out = Out_;
  using In = In_;
  using W = W_;
  static constexpr int lanes = lanes_;

  Out* __restrict__ out;
  const In* __restrict__ in;
  const W* __restrict__ filter;
  ShapeDesc shape;
};

/** @defgroup ConvolutionROI Describes the output shape and how it maps into the input.
 * @{
 */

struct ROI {
  int h_begin, h_end, wc_begin, wc_end;
  int h, wc;
  int64_t hwc;
};

/**
 * @brief Creates ROI description for input and output of the same shape.
 */
struct ROIFull {
  DALI_DEVICE DALI_FORCEINLINE ROI operator()(const ShapeDesc& shape_desc) {
    return {0, shape_desc.h, 0, shape_desc.wc, shape_desc.h, shape_desc.wc, shape_desc.hwc};
  }
};

/**
 * @brief Creates (maximal) ROI shrunk so that all filter positions lie fully within the image.
 */
struct ROIOnlyValid {
  DALI_DEVICE DALI_FORCEINLINE ROI operator()(const ShapeDesc& shape_desc) {
    int h_begin = -shape_desc.filter_top_anchor;
    int h_end = shape_desc.h - (shape_desc.r - 1 + shape_desc.filter_top_anchor);
    int wc_begin = (-shape_desc.filter_left_anchor) * shape_desc.c;
    int wc_end = shape_desc.wc - (shape_desc.s - 1 + shape_desc.filter_left_anchor) * shape_desc.c;
    int h = h_end - h_begin;
    int wc = wc_end - wc_begin;
    return {h_begin, h_end, wc_begin, wc_end, h, wc, h * wc};
  }
};

/** @} */  // end of ConvolutionROI


/** @defgroup InputLoader InputLoader is meant to specialize loading of the sample from global
 * memory. This is how different border modes (apart from BORDER_VALID) are handled.
 * @{
 */

template <bool degenerated_extents>
struct Reflect101 {
  DALI_HOST_DEV DALI_FORCEINLINE int border_remap(int idx, int len) const {
    assert(len > 0);
    return boundary::idx_reflect_101(idx, len);
  }
};

struct Reflect1001 {
  DALI_HOST_DEV DALI_FORCEINLINE int border_remap(int idx, int len) const {
    assert(len > 0);
    return boundary::idx_reflect_1001(idx, len);
  }
};

struct Clamp {
  DALI_HOST_DEV DALI_FORCEINLINE int border_remap(int idx, int len) const {
    assert(len > 0);
    return boundary::idx_clamp(idx, 0, len);
  }
};

struct Wrap {
  DALI_HOST_DEV DALI_FORCEINLINE int border_remap(int idx, int len) const {
    assert(len > 0);
    return boundary::idx_wrap(idx, len);
  }
};

template <typename Remap, typename In>
struct InLoaderBorderRemap : protected Remap {
  DALI_HOST_DEV DALI_FORCEINLINE int remap_height(int idx, const ShapeDesc& sample_shape) const {
    return this->border_remap(idx, sample_shape.h);
  }

  DALI_HOST_DEV DALI_FORCEINLINE int remap_width(int idx, const ShapeDesc& sample_shape) const {
    return border_remap_strided(idx, sample_shape.w, sample_shape.c, sample_shape.wc);
  }

  DALI_HOST_DEV DALI_FORCEINLINE In load(const In* __restrict__ in, int y, int x,
                                         const ShapeDesc& sample_shape) const {
    return in[y * static_cast<int64_t>(sample_shape.wc) + x];
  }

 protected:
  DALI_HOST_DEV DALI_FORCEINLINE int border_remap_strided(int idx, int reflect_dim_size,
                                                          int inner_stride,
                                                          int total_stride) const {
    if (idx < 0) {
      int reflect_dim_idx = (idx + 1) / inner_stride - 1;
      int inner_dim_idx = (idx + 1) % inner_stride + inner_stride - 1;
      return this->border_remap(reflect_dim_idx, reflect_dim_size) * inner_stride + inner_dim_idx;
    }
    if (idx >= total_stride) {
      return this->border_remap(idx / inner_stride, reflect_dim_size) * inner_stride +
             idx % inner_stride;
    }
    return idx;
  }
};

template <typename In>
struct InLoaderPad {
  DALI_HOST_DEV DALI_FORCEINLINE int remap_height(int idx, const ShapeDesc& sample_shape) const {
    return idx;
  }

  DALI_HOST_DEV DALI_FORCEINLINE int remap_width(int idx, const ShapeDesc& sample_shape) const {
    return idx;
  }

  DALI_HOST_DEV DALI_FORCEINLINE In load(const In* __restrict__ in, int y, int x,
                                         const ShapeDesc& sample_shape) const {
    if (y < 0 || x < 0 || x >= sample_shape.wc || y >= sample_shape.h) {
      return fill_value;
    }
    return in[y * static_cast<int64_t>(sample_shape.wc) + x];
  }

  In fill_value;
};

template <typename InLoader>
struct InLoaderFactory {
  using T = InLoader;
  DALI_DEVICE DALI_FORCEINLINE T& operator()(int sample_idx) {
    return in_loader;
  }
  T in_loader;
};

template <typename In>
struct InLoaderFactory<InLoaderPad<In>> {
  using T = InLoaderPad<In>;
  DALI_DEVICE DALI_FORCEINLINE T operator()(int sample_idx) {
    if (fill_values == nullptr) {
      return {0};
    }
    return {fill_values[sample_idx][0]};
  }
  const In** fill_values;
};

/** @} */  // end of InputLoader


/** @defgroup InputConv The specializations provide ``compute`` method that computes convolution
 * for the ``block_width x lanes`` patch and stores it in the provided ``acc``.
 * @{
 */

/**
 * @brief First loads the input roi neceesary to compute the output of shape ``block_width x
 * lanes`` into shared memory (including filter's halo/apron), then computes the convolution.
 */
template <typename SampleDescT, typename Inloader>
struct ShmInputConv {
  using In = typename SampleDescT::In;
  using Acc = typename SampleDescT::Acc;

  DALI_DEVICE DALI_FORCEINLINE ShmInputConv(const SampleDescT& sample_desc,
                                            const Inloader& in_loader, In* in_workspace)
      : sample_desc{sample_desc}, in_loader{in_loader}, in_workspace{in_workspace} {}

  DALI_DEVICE DALI_FORCEINLINE void compute(Acc* __restrict__ acc, const In* __restrict__ in,
                                            int y_start, int x_start) const {
    __syncthreads();
    load_input_to_shm(in, y_start, x_start);
    __syncthreads();
    const auto* filter = sample_desc.filter;
    for (int s = 0; s < sample_desc.shape.s; s++) {
      int x = threadIdx.x + s * sample_desc.shape.c;
      for (int r = 0; r < sample_desc.shape.r; r++) {
        auto filter_coef = __ldg(filter + r * sample_desc.shape.s + s);
#pragma unroll
        for (int lane = 0; lane < SampleDescT::lanes; lane++) {
          int y = lane + r;
          auto in_val = in_workspace[y * sample_desc.shape.in_workspace_width + x];
          acc[lane] += in_val * filter_coef;
        }
      }
    }
  }

  DALI_DEVICE DALI_FORCEINLINE void load_input_to_shm(const In* __restrict__ in, int y_start,
                                                      int x_start) const {
    for (int x = threadIdx.x; x < sample_desc.shape.in_workspace_width; x += blockDim.x) {
      auto global_x = in_loader.remap_width(
          x_start + x + sample_desc.shape.filter_left_anchor * sample_desc.shape.c,
          sample_desc.shape);
      auto load_row = [&](int y) {
        int global_y = in_loader.remap_height(y_start + y + sample_desc.shape.filter_top_anchor,
                                              sample_desc.shape);
        in_workspace[y * sample_desc.shape.in_workspace_width + x] =
            in_loader.load(in, global_y, global_x, sample_desc.shape);
      };
#pragma unroll
      for (int y = 0; y < SampleDescT::lanes; y++) {
        load_row(y);
      }
      for (int y = SampleDescT::lanes; y < SampleDescT::lanes + sample_desc.shape.r - 1; y++) {
        load_row(y);
      }
    }
  }

  const SampleDescT& sample_desc;
  const Inloader& in_loader;
  In* in_workspace;
};

/**
 * @brief Computes the convolution of size ``block_width x lanes`` accessing the input directly in
 * global memory. Used as a fallback when the filter size of number of channels in the input makes
 * it impossible to use ``ShmInputConv``.
 */
template <typename SampleDescT, typename Inloader>
struct DirectInputConv {
  using Acc = typename SampleDescT::Acc;
  using In = typename SampleDescT::In;

  DALI_DEVICE DALI_FORCEINLINE DirectInputConv(const SampleDescT& sample_desc,
                                               const Inloader& in_loader)
      : sample_desc{sample_desc}, in_loader{in_loader} {}

  DALI_DEVICE DALI_FORCEINLINE void compute(Acc* __restrict__ acc, const In* __restrict__ in,
                                            int y_start, int x_start) const {
    const auto* filter = sample_desc.filter;
    for (int s = 0; s < sample_desc.shape.s; s++) {
      auto global_x = in_loader.remap_width(
          x_start + threadIdx.x + (sample_desc.shape.filter_left_anchor + s) * sample_desc.shape.c,
          sample_desc.shape);
      for (int r = 0; r < sample_desc.shape.r; r++) {
        auto filter_coef = __ldg(filter + r * sample_desc.shape.s + s);
        // Even without shm, using `lanes` speeds up the kernel by reducing
        // the cost of nested loops arithmetic per single output value
#pragma unroll
        for (int lane = 0; lane < SampleDescT::lanes; lane++) {
          auto global_y = in_loader.remap_height(
              y_start + lane + r + sample_desc.shape.filter_top_anchor, sample_desc.shape);
          auto in_val = in_loader.load(in, global_y, global_x, sample_desc.shape);
          acc[lane] += in_val * filter_coef;
        }
      }
    }
  }

  const SampleDescT& sample_desc;
  const Inloader& in_loader;
};

/** @} */  // end of InputConv

template <int lanes, typename Out, typename Acc>
DALI_DEVICE DALI_FORCEINLINE void store_acc_in_global_output(Out* __restrict__ out,
                                                             const Acc* __restrict__ acc,
                                                             const ROI& roi, int y_start,
                                                             int x_start) {
  int x = x_start + threadIdx.x;
  if (x < roi.wc_end) {
#pragma unroll
    for (int lane = 0; lane < lanes; lane++) {
      int y = y_start + lane;
      if (y < roi.h_end) {
        out[(y - roi.h_begin) * static_cast<int64_t>(roi.wc) + (x - roi.wc_begin)] =
            ConvertSat<Out>(acc[lane]);
      }
    }
  }
}

template <typename SampleDescT, typename Conv>
DALI_DEVICE DALI_FORCEINLINE void stride_grid(const SampleDescT& sample_desc, const Conv& conv,
                                              const ROI roi) {
  constexpr int lanes = SampleDescT::lanes;
  const auto* in = sample_desc.in;
  auto* out = sample_desc.out;
  for (int f = 0; f < sample_desc.shape.f; f++, in += sample_desc.shape.hwc, out += roi.hwc) {
    for (int y_start = lanes * blockIdx.y + roi.h_begin; y_start < roi.h_end;
         y_start += gridDim.y * lanes) {
      for (int x_start = blockDim.x * blockIdx.x + roi.wc_begin; x_start < roi.wc_end;
           x_start += gridDim.x * blockDim.x) {
        typename SampleDescT::Acc acc[lanes] = {};
        conv.compute(acc, in, y_start, x_start);
        store_acc_in_global_output<lanes>(out, acc, roi, y_start, x_start);
      }
    }
  }
}

template <typename SampleDescT, typename InLoaderFactory, typename ROIFactory>
__global__ void filter2d(const SampleDescT* __restrict__ descs, InLoaderFactory in_loader_factory,
                         ROIFactory roi_factory) {
  using In = typename SampleDescT::In;
  using InLoader = typename InLoaderFactory::T;
  extern __shared__ char shm[];
  auto sample_desc = descs[blockIdx.z];
  auto&& in_loader = in_loader_factory(blockIdx.z);
  ROI roi = roi_factory(sample_desc.shape);
  if (sample_desc.shape.in_workspace_width) {
    In* in_workspace = reinterpret_cast<In*>(shm);
    ShmInputConv<SampleDescT, InLoader> conv{sample_desc, in_loader, in_workspace};
    stride_grid(sample_desc, conv, roi);
  } else {
    DirectInputConv<SampleDescT, InLoader> conv{sample_desc, in_loader};
    stride_grid(sample_desc, conv, roi);
  }
}
}  // namespace filter

template <typename Out, typename In, typename W, bool has_channel_dim, bool has_sequence_dim>
struct Filter2dGpu {
  /* It computes a corellation of the input and the filter.
  Flip filter in both dimensions for a convolution. */

  static constexpr int axes = 2;
  static constexpr int num_sequence_dim = static_cast<int>(has_sequence_dim);
  static constexpr int num_channels_dim = static_cast<int>(has_channel_dim);
  static constexpr int ndim = num_sequence_dim + axes + num_channels_dim;
  static constexpr int filter_ndim = axes;
  using Intermediate = decltype(std::declval<W>() * std::declval<In>());

  static constexpr int block_width = 128;
  static constexpr int lanes = 8;
  static constexpr int max_grid_height = 32;
  static constexpr int max_grid_width = 32;
  static constexpr int max_grid_rows = max_grid_height * lanes;
  static constexpr int max_grid_cols = max_grid_width * block_width;
  static constexpr int max_sample_height =
      std::numeric_limits<int>::max() / max_grid_rows * max_grid_rows;
  static constexpr int max_sample_width =
      std::numeric_limits<int>::max() / max_grid_cols * max_grid_cols;

  using SampleDescT = filter::SampleDesc<Out, In, W, Intermediate, lanes>;

  void Run(KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim>& out,
           const TensorListView<StorageGPU, const In, ndim>& in,
           const TensorListView<StorageGPU, const W, axes>& filters,
           const TensorListView<StorageCPU, const int, 1>& anchors,
           boundary::BoundaryType border_type,
           const TensorListView<StorageGPU, const In, 0>& fill_values = {}) {
    auto num_samples = in.shape.num_samples();

    samples_desc_.clear();
    samples_desc_.reserve(num_samples);
    const auto& in_shapes = in.shape;
    const auto& filter_shapes = filters.shape;

    int shared_mem_limit = GetSharedMemPerBlock();
    int max_total_workspace = 0;
    bool any_has_degenerated_extents = false;
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      const auto& in_shape = in_shapes[sample_idx];
      const auto& filter_shape = filter_shapes[sample_idx];
      const auto& anchor_view = anchors[sample_idx];
      assert(anchor_view.shape.num_elements() == filter_ndim);
      int required_workspace;
      bool has_degenerated_extents;
      auto shape_desc =
          SetupSampleShapeDesc(required_workspace, has_degenerated_extents, sample_idx, in_shape,
                               filter_shape, anchor_view, shared_mem_limit);
      any_has_degenerated_extents |= has_degenerated_extents;
      max_total_workspace = std::max(max_total_workspace, required_workspace);
      samples_desc_.push_back({out.tensor_data(sample_idx), in.tensor_data(sample_idx),
                               filters.tensor_data(sample_idx), shape_desc});
    }
    int max_width = 0, max_height = 0;
    ComputeExtentsMax(max_height, max_width, out.shape);
    SampleDescT* descs_dev;
    std::tie(descs_dev) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, samples_desc_);
    int num_blocks_h = div_ceil(max_height, lanes);
    int num_blocks_w = div_ceil(max_width, block_width);
    num_blocks_h = std::min(num_blocks_h, max_grid_height);
    num_blocks_w = std::min(num_blocks_w, max_grid_width);
    dim3 grid(num_blocks_w, num_blocks_h, num_samples);
    dim3 block(block_width, 1, 1);
    RunKernelWithBorderMode(
        ctx, border_type, fill_values, any_has_degenerated_extents,
        [&](auto&& loader, auto&& roi_factory) {
          filter::filter2d<<<grid, block, max_total_workspace, ctx.gpu.stream>>>(descs_dev, loader,
                                                                                 roi_factory);
          CUDA_CALL(cudaGetLastError());
        });
  }

 protected:
  template <typename KernelLauncher>
  void RunKernelWithBorderMode(KernelContext& ctx, boundary::BoundaryType border_type,
                               const TensorListView<StorageGPU, const In, 0>& fill_values,
                               bool has_degenerated_extents, KernelLauncher&& launch_kernel) {
    using namespace boundary;  // NOLINT(build/namespaces)
    switch (border_type) {
      case BoundaryType::REFLECT_101:
        RunKernelBorder101(ctx, has_degenerated_extents, std::move(launch_kernel));
        break;
      case BoundaryType::REFLECT_1001:
        RunKernelBorderRemap<filter::ROIFull, filter::Reflect1001>(ctx, std::move(launch_kernel));
        break;
      case BoundaryType::CLAMP:
        RunKernelBorderRemap<filter::ROIFull, filter::Clamp>(ctx, std::move(launch_kernel));
        break;
      case BoundaryType::WRAP:
        RunKernelBorderRemap<filter::ROIFull, filter::Wrap>(ctx, std::move(launch_kernel));
        break;
      case BoundaryType::CONSTANT:
        RunKernelBorderConstant(ctx, fill_values, std::move(launch_kernel));
        break;
      case BoundaryType::ISOLATED:
        RunKernelBorderRemap<filter::ROIOnlyValid>(ctx, std::move(launch_kernel));
        break;
      default:
        DALI_FAIL(
            make_string("Unsupported border type was specified: ", to_string(border_type), "."));
    }
  }

  template <typename ROIFactory, typename Remap = filter::Reflect1001, typename KernelLauncher>
  void RunKernelBorderRemap(KernelContext& ctx, KernelLauncher&& launch_kernel) {
    using Loader = filter::InLoaderBorderRemap<Remap, In>;
    filter::InLoaderFactory<Loader> loader_factory{Loader{}};
    launch_kernel(std::move(loader_factory), ROIFactory{});
  }

  template <typename KernelLauncher>
  void RunKernelBorder101(KernelContext& ctx, bool has_degenerated_extents,
                          KernelLauncher&& launch_kernel) {
    // If any of the samples has some extent equal to 1, border handler needs extra
    // check to prevent infinite loop. Extra check for every single position of the filter
    // over an image is costly, so try to avoid it.
    BOOL_SWITCH(
        has_degenerated_extents, HasDegeneratedExtents,
        (using Loader = filter::InLoaderBorderRemap<filter::Reflect101<HasDegeneratedExtents>, In>;
         filter::InLoaderFactory<Loader> loader_factory{Loader{}};
         launch_kernel(std::move(loader_factory), filter::ROIFull{});));  // NOLINT
  }

  template <typename KernelLauncher>
  void RunKernelBorderConstant(KernelContext& ctx,
                               const TensorListView<StorageGPU, const In, 0>& fill_values,
                               KernelLauncher&& launch_kernel) {
    int num_samples = samples_desc_.size();
    assert(fill_values.num_samples() == num_samples || fill_values.num_samples() == 0);
    if (fill_values.num_samples() != num_samples) {
      filter::InLoaderFactory<filter::InLoaderPad<In>> loader_factory{nullptr};
      launch_kernel(std::move(loader_factory), filter::ROIFull{});
    } else {
      fill_values_.resize(num_samples);
      for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
        fill_values_[sample_idx] = fill_values[sample_idx].data;
      }
      const In** fill_values_dev;
      std::tie(fill_values_dev) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, fill_values_);
      filter::InLoaderFactory<filter::InLoaderPad<In>> loader_factory{fill_values_dev};
      launch_kernel(std::move(loader_factory), filter::ROIFull{});
    }
  }

  template <typename OutShapes>
  void ComputeExtentsMax(int& max_height, int& max_width, const OutShapes& out_shapes) {
    for (int sample_idx = 0; sample_idx < out_shapes.num_samples(); sample_idx++) {
      const auto& out_shape = out_shapes[sample_idx];
      // Assuming those are no greater than the in_shapes,
      // overflow limits were already checked for the input
      int h = out_shape[num_sequence_dim];
      int w = out_shape[num_sequence_dim + 1];
      int c = has_channel_dim ? out_shape[num_sequence_dim + 2] : 1;
      max_height = std::max(max_height, h);
      max_width = std::max(max_width, w * c);
    }
  }

  template <typename InShape, typename FilterShape, typename AnchorView>
  filter::ShapeDesc SetupSampleShapeDesc(int& required_worskapce, bool& has_degenerated_extents,
                                         int sample_idx, const InShape& in_shape,
                                         const FilterShape& filter_shape, const AnchorView& anchor,
                                         int shared_mem_limit) {
    auto filter_vol = volume(filter_shape);
    auto r = filter_shape[0];
    auto s = filter_shape[1];
    auto filter_top_anchor = anchor.data[0] == -1 ? r / 2 : anchor.data[0];
    auto filter_left_anchor = anchor.data[1] == -1 ? s / 2 : anchor.data[1];
    DALI_ENFORCE(
        0 <= filter_top_anchor && filter_top_anchor < r && 0 <= filter_left_anchor &&
            filter_left_anchor < s,
        make_string("Anchor must lie within the filter. Got anchor ",
                    TensorShape<2>{filter_top_anchor, filter_left_anchor}, " with filter of shape ",
                    filter_shape, " for sample of idx ", sample_idx, "."));
    filter_top_anchor = -filter_top_anchor;
    filter_left_anchor = -filter_left_anchor;
    auto f = has_sequence_dim ? in_shape[0] : 1;
    auto h = in_shape[num_sequence_dim];
    auto w = in_shape[num_sequence_dim + 1];
    auto c = has_channel_dim ? in_shape[num_sequence_dim + 2] : 1;
    auto wc = w * c;
    auto hwc = h * wc;
    has_degenerated_extents = h == 1 || w == 1;
    ValidateSampleNumericLimits(sample_idx, r, s, filter_vol, filter_top_anchor, filter_left_anchor,
                                f, h, wc, c);
    auto in_workspace_width = block_width + (s - 1) * c;
    auto in_workspace_num_elements = in_workspace_width * (lanes + r - 1);
    if (in_workspace_width > std::numeric_limits<int>::max() ||
        in_workspace_num_elements > std::numeric_limits<int>::max()) {
      in_workspace_width = in_workspace_num_elements = 0;
    }
    required_worskapce = in_workspace_num_elements * sizeof(In);
    if (c > block_width || required_worskapce > shared_mem_limit) {
      required_worskapce = in_workspace_width = 0;
    }
    return {hwc,
            static_cast<int>(wc),
            static_cast<int>(f),
            static_cast<int>(h),
            static_cast<int>(w),
            static_cast<int>(c),
            static_cast<int>(filter_vol),
            static_cast<int>(r),
            static_cast<int>(s),
            static_cast<int>(filter_top_anchor),
            static_cast<int>(filter_left_anchor),
            static_cast<int>(in_workspace_width)};
  }

  void ValidateSampleNumericLimits(int sample_idx, int64_t r, int64_t s, int64_t filter_vol,
                                   int64_t filter_top_anchor, int64_t filter_left_anchor, int64_t f,
                                   int64_t h, int64_t wc, int64_t c) {
    DALI_ENFORCE(
        filter_vol <= std::numeric_limits<int>::max(),
        make_string("Volume of filter for sample of idx ", sample_idx, " exceedes the limit of ",
                    std::numeric_limits<int>::max(), ". Got: ", filter_vol, "."));
    DALI_ENFORCE(
        f <= std::numeric_limits<int>::max(),
        make_string("Number of frames for sample of idx ", sample_idx, " exceedes the limit of ",
                    std::numeric_limits<int>::max(), ". Got: ", f, "."));
    DALI_ENFORCE(h <= max_sample_height,
                 make_string("The height of sample of idx ", sample_idx, " exceedes the limit of ",
                             max_sample_height, ". Got: ", h, "."));
    DALI_ENFORCE(0 <= wc && wc <= max_sample_width,
                 make_string("The total width and number of channels in sample of idx ", sample_idx,
                             " exceedes the limit of ", max_sample_width, ". Got: ", wc, "."));
    auto height_radious = h + r + filter_top_anchor - 2;
    DALI_ENFORCE(
        0 <= height_radious && height_radious <= std::numeric_limits<int>::max(),
        make_string("The combined height of the sample and filter radious for sample of idx ",
                    sample_idx, " exceedes the limit of ", std::numeric_limits<int>::max(),
                    ". Got: ", height_radious, "."));
    auto width_radious = wc - 1 + (s - 1 + filter_left_anchor) * c;
    DALI_ENFORCE(
        0 <= width_radious && width_radious <= std::numeric_limits<int>::max(),
        make_string("The combined width, number of channels and filter radious for sample of idx ",
                    sample_idx, " exceedes the limit of ", std::numeric_limits<int>::max(),
                    ". Got: ", width_radious, "."));
  }

  std::vector<SampleDescT> samples_desc_;
  std::vector<const In*> fill_values_;
};


// WAR c++14 odr usage issue (make_string in error message takes them as l-values)
// it should be unnecessary in c++17
template <typename Out, typename In, typename W, bool has_channel_dim, bool has_sequence_dim>
constexpr int Filter2dGpu<Out, In, W, has_channel_dim, has_sequence_dim>::max_sample_height;

template <typename Out, typename In, typename W, bool has_channel_dim, bool has_sequence_dim>
constexpr int Filter2dGpu<Out, In, W, has_channel_dim, has_sequence_dim>::max_sample_width;

template <typename Out, typename In, typename W, bool has_channel_dim, bool has_sequence_dim>
constexpr int Filter2dGpu<Out, In, W, has_channel_dim, has_sequence_dim>::max_grid_height;

template <typename Out, typename In, typename W, bool has_channel_dim, bool has_sequence_dim>
constexpr int Filter2dGpu<Out, In, W, has_channel_dim, has_sequence_dim>::max_grid_width;

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_FILTER_GPU_H_
