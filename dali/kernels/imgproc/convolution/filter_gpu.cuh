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
#include "dali/core/geom/vec.h"
#include "dali/core/span.h"
#include "dali/core/static_switch.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

namespace filter {

// Descrption of the input ROI passed to the kernel
template <int axes>
struct InputROI {
  ivec<axes> begin, end;
};

template <int axes>
struct LogicalBlock {
  // must be power of 2
  ivec<axes> extents_log2;
  ivec<axes> strides_log2;

  DALI_HOST_DEV DALI_FORCEINLINE ivec<axes> lBlockDim() const {
    auto block = 1 << extents_log2;
    assert(blockDim.x * blockDim.y * blockDim.z == volume(block));
    return block;
  }

  DALI_DEVICE DALI_FORCEINLINE ivec<axes> lThreadIdx() const {
    assert(threadIdx.y == 0);
    assert(threadIdx.z == 0);
    return (int(threadIdx.x) >> strides_log2) & (lBlockDim() - 1);
  }
};

inline LogicalBlock<2> create_log_block(ivec2 xy_log2) {
  return {xy_log2, {0, xy_log2.x}};
}

template <int axes>
struct ShapeDesc {
  int64_t hwc;
  int wc, f, h, w, c;
  ivec<axes> filter_extents;
  ivec<axes> anchor_shift;
  ivec<axes> workspace_extents;
  LogicalBlock<axes> log_block;
};

template <typename Out_, typename In_, typename W_, typename Acc_, int axes_, int lanes_>
struct SampleDesc {
  using Acc = Acc_;
  using Out = Out_;
  using In = In_;
  using W = W_;
  static constexpr int lanes = lanes_;
  static constexpr int axes = axes_;

  Out* __restrict__ out;
  const In* __restrict__ in;
  const W* __restrict__ filter;
  ShapeDesc<axes> shape;
};

// Internal representation of input ROI for cuda kernel.
struct InputROIDesc {
  int h_begin, h_end, wc_begin, wc_end;
  int h, wc;
  int64_t hwc;
};

template <int axes>
struct InputRoiFull {
  DALI_DEVICE DALI_FORCEINLINE InputROIDesc operator()(const ShapeDesc<axes>& shape_desc,
                                                       int sample_idx) {
    (void)sample_idx;
    return {0, shape_desc.h, 0, shape_desc.wc, shape_desc.h, shape_desc.wc, shape_desc.hwc};
  }
};

template <int axes>
struct CustomInputROI {
  DALI_DEVICE DALI_FORCEINLINE InputROIDesc operator()(const ShapeDesc<axes>& shape_desc,
                                                       int sample_idx) {
    auto roi = rois[sample_idx];
    roi.begin[axes - 1] *= shape_desc.c;
    roi.end[axes - 1] *= shape_desc.c;
    auto size = roi.end - roi.begin;
    return {roi.begin[0], roi.end[0], roi.begin[1],     roi.end[1],
            size[0],      size[1],    size[0] * size[1]};
  }

  const InputROI<axes>* rois;
};

/** @defgroup InputLoader InputLoader is meant to specialize loading of the sample from global
 * memory. This is how different border modes (apart from BORDER_VALID) are handled.
 * @{
 */

template <bool degenerated_extents>
struct Reflect101 {
  DALI_HOST_DEV DALI_FORCEINLINE int border_remap(int idx, int len) const {
    assert(len > 0);
    return boundary::idx_reflect_101<int, degenerated_extents>(idx, len);
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

template <typename Remap, typename In, int axes>
struct InLoaderBorderRemap : protected Remap {
  DALI_HOST_DEV DALI_FORCEINLINE int remap_height(int idx,
                                                  const ShapeDesc<axes>& sample_shape) const {
    return this->border_remap(idx, sample_shape.h);
  }

  DALI_HOST_DEV DALI_FORCEINLINE int remap_width(int idx,
                                                 const ShapeDesc<axes>& sample_shape) const {
    return border_remap_strided(idx, sample_shape.w, sample_shape.c, sample_shape.wc);
  }

  DALI_HOST_DEV DALI_FORCEINLINE In load(const In* __restrict__ in, int y, int x,
                                         const ShapeDesc<axes>& sample_shape) const {
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

template <typename In, int axes>
struct InLoaderPad {
  DALI_HOST_DEV DALI_FORCEINLINE int remap_height(int idx,
                                                  const ShapeDesc<axes>& sample_shape) const {
    return idx;
  }

  DALI_HOST_DEV DALI_FORCEINLINE int remap_width(int idx,
                                                 const ShapeDesc<axes>& sample_shape) const {
    return idx;
  }

  DALI_HOST_DEV DALI_FORCEINLINE In load(const In* __restrict__ in, int y, int x,
                                         const ShapeDesc<axes>& sample_shape) const {
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

template <typename In, int axes>
struct InLoaderFactory<InLoaderPad<In, axes>> {
  using T = InLoaderPad<In, axes>;
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
    const auto& log_block = sample_desc.shape.log_block;
    const auto& lBlockDim = log_block.lBlockDim();
    const auto& lThreadIdx = log_block.lThreadIdx();
    __syncthreads();
    load_input_to_shm(in, y_start, x_start);
    __syncthreads();
    const auto* filter = sample_desc.filter;
    for (int s = 0; s < sample_desc.shape.filter_extents[1]; s++) {
      int x = lThreadIdx.x + s * sample_desc.shape.c;
      for (int r = 0; r < sample_desc.shape.filter_extents[0]; r++) {
        auto filter_coef = __ldg(filter + r * sample_desc.shape.filter_extents[1] + s);
#pragma unroll
        for (int lane = 0, lanes_offset = 0; lane < SampleDescT::lanes;
             lane++, lanes_offset += lBlockDim.y) {
          int y = lThreadIdx.y + r + lanes_offset;
          auto in_val = in_workspace[y * sample_desc.shape.workspace_extents[1] + x];
          acc[lane] += in_val * filter_coef;
        }
      }
    }
  }

  DALI_DEVICE DALI_FORCEINLINE void load_input_to_shm(const In* __restrict__ in, int y_start,
                                                      int x_start) const {
    const auto& log_block = sample_desc.shape.log_block;
    const auto& lBlockDim = log_block.lBlockDim();
    const auto& lThreadIdx = log_block.lThreadIdx();
    x_start -= sample_desc.shape.anchor_shift[1];
    y_start -= sample_desc.shape.anchor_shift[0];
    for (int x = lThreadIdx.x; x < sample_desc.shape.workspace_extents[1]; x += lBlockDim.x) {
      int global_x = in_loader.remap_width(x_start + x, sample_desc.shape);
#pragma unroll SampleDescT::lanes
      for (int y = lThreadIdx.y; y < sample_desc.shape.workspace_extents[0]; y += lBlockDim.y) {
        int global_y = in_loader.remap_height(y_start + y, sample_desc.shape);
        in_workspace[y * sample_desc.shape.workspace_extents[1] + x] =
            in_loader.load(in, global_y, global_x, sample_desc.shape);
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
    const auto& log_block = sample_desc.shape.log_block;
    const auto& lBlockDim = log_block.lBlockDim();
    const auto& lThreadIdx = log_block.lThreadIdx();
    const auto* filter = sample_desc.filter;
    for (int s = 0; s < sample_desc.shape.filter_extents[1]; s++) {
      auto global_x = in_loader.remap_width(
          x_start + lThreadIdx.x + s * sample_desc.shape.c - sample_desc.shape.anchor_shift[1],
          sample_desc.shape);
      for (int r = 0; r < sample_desc.shape.filter_extents[0]; r++) {
        auto filter_coef = __ldg(filter + r * sample_desc.shape.filter_extents[1] + s);
        // Even without shm, using `lanes` speeds up the kernel by reducing
        // the cost of nested loops arithmetic per single output value
#pragma unroll
        for (int lane = 0; lane < SampleDescT::lanes; lane++) {
          auto global_y = in_loader.remap_height(
              y_start + lThreadIdx.y + lane * lBlockDim.y + r - sample_desc.shape.anchor_shift[0],
              sample_desc.shape);
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

template <int lanes, typename Out, typename Acc, int axes>
DALI_DEVICE DALI_FORCEINLINE void store_acc_in_global_output(Out* __restrict__ out,
                                                             const Acc* __restrict__ acc,
                                                             const InputROIDesc& roi, int y_start,
                                                             int x_start,
                                                             const LogicalBlock<axes>& log_block) {
  // The x, y indices describe the input, they are mapped into, output space by shifting
  // by roi.h_begin/wc_begin. Moreover the size of input roi is assumed to be equal to
  // the output size, so its extents are used to prevent OOB accesses
  // (due to grid/lanes protruding at the boundary).
  const auto& lBlockDim = log_block.lBlockDim();
  const auto& lThreadIdx = log_block.lThreadIdx();
  int x = x_start + lThreadIdx.x;
  if (x < roi.wc_end) {
#pragma unroll
    for (int lane = 0; lane < lanes; lane++) {
      int y = y_start + lThreadIdx.y + lane * lBlockDim.y;
      if (y < roi.h_end) {
        out[(y - roi.h_begin) * static_cast<int64_t>(roi.wc) + (x - roi.wc_begin)] =
            ConvertSat<Out>(acc[lane]);
      }
    }
  }
}

template <typename SampleDescT, typename Conv>
DALI_DEVICE DALI_FORCEINLINE void stride_grid(const SampleDescT& sample_desc,
                                              const InputROIDesc& roi, const Conv& conv) {
  constexpr int lanes = SampleDescT::lanes;
  const auto* in = sample_desc.in;
  auto* out = sample_desc.out;
  const auto& log_block = sample_desc.shape.log_block;
  const auto& lBlockDim = log_block.lBlockDim();
  for (int f = 0; f < sample_desc.shape.f; f++, in += sample_desc.shape.hwc, out += roi.hwc) {
    for (int y_start = lanes * lBlockDim.y * blockIdx.y + roi.h_begin; y_start < roi.h_end;
         y_start += lanes * lBlockDim.y * gridDim.y) {
      for (int x_start = lBlockDim.x * blockIdx.x + roi.wc_begin; x_start < roi.wc_end;
           x_start += gridDim.x * lBlockDim.x) {
        typename SampleDescT::Acc acc[lanes] = {};
        conv.compute(acc, in, y_start, x_start);
        store_acc_in_global_output<lanes>(out, acc, roi, y_start, x_start, log_block);
      }
    }
  }
}

template <typename SampleDescT, typename InputROIFactory, typename InLoaderFactory>
__global__ void filter2d(const SampleDescT* __restrict__ descs, InputROIFactory in_roi_factory,
                         InLoaderFactory in_loader_factory) {
  using In = typename SampleDescT::In;
  using InLoader = typename InLoaderFactory::T;
  extern __shared__ char shm[];
  auto sample_desc = descs[blockIdx.z];
  auto&& roi = in_roi_factory(sample_desc.shape, blockIdx.z);
  auto&& in_loader = in_loader_factory(blockIdx.z);
  if (sample_desc.shape.workspace_extents[1]) {
    In* in_workspace = reinterpret_cast<In*>(shm);
    ShmInputConv<SampleDescT, InLoader> conv{sample_desc, in_loader, in_workspace};
    stride_grid(sample_desc, roi, conv);
  } else {
    DirectInputConv<SampleDescT, InLoader> conv{sample_desc, in_loader};
    stride_grid(sample_desc, roi, conv);
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

  using SampleDescT = filter::SampleDesc<Out, In, W, Intermediate, axes, lanes>;

  void Run(KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim>& out,
           const TensorListView<StorageGPU, const In, ndim>& in,
           const TensorListView<StorageGPU, const W, axes>& filters,
           const span<const ivec<axes>> anchors, boundary::BoundaryType border_type,
           const span<const filter::InputROI<axes>> input_rois = {},
           const TensorListView<StorageGPU, const In, 0>& fill_values = {}) {
    auto num_samples = in.shape.num_samples();
    assert(out.num_samples() == num_samples && filters.num_samples() == num_samples &&
           anchors.size() == num_samples);
    assert(fill_values.num_samples() == num_samples || fill_values.num_samples() == 0);
    assert(input_rois.size() == num_samples || input_rois.size() == 0);

    samples_desc_.clear();
    samples_desc_.reserve(num_samples);
    const auto& in_shapes = in.shape;
    const auto& out_shapes = out.shape;
    const auto& filter_shapes = filters.shape;

    int shared_mem_limit = GetSharedMemPerBlock();
    int max_total_workspace = 0;
    bool any_has_degenerated_extents = false;
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      const auto& in_shape = in_shapes[sample_idx];
      const auto& filter_shape = filter_shapes[sample_idx];
      int required_workspace;
      bool has_degenerated_extents;
      auto shape_desc =
          SetupSampleShapeDesc(required_workspace, has_degenerated_extents, sample_idx, in_shape,
                               filter_shape, anchors[sample_idx], shared_mem_limit);
      any_has_degenerated_extents |= has_degenerated_extents;
      max_total_workspace = std::max(max_total_workspace, required_workspace);
      samples_desc_.push_back({out.tensor_data(sample_idx), in.tensor_data(sample_idx),
                               filters.tensor_data(sample_idx), shape_desc});
    }
    SampleDescT* descs_dev;
    std::tie(descs_dev) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, samples_desc_);
    int max_width = 0, max_height = 0;
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      const auto& out_shape = out_shapes[sample_idx];
      int h = out_shape[num_sequence_dim];
      int w = out_shape[num_sequence_dim + 1];
      int c = has_channel_dim ? out_shape[num_sequence_dim + 2] : 1;
      max_height = std::max(max_height, h);
      max_width = std::max(max_width, w * c);
    }
    int num_blocks_h = div_ceil(max_height, lanes);
    int num_blocks_w = div_ceil(max_width, block_width);
    num_blocks_h = std::min(num_blocks_h, max_grid_height);
    num_blocks_w = std::min(num_blocks_w, max_grid_width);
    dim3 grid(num_blocks_w, num_blocks_h, num_samples);
    dim3 block(block_width, 1, 1);
    RunKernel(ctx, out_shapes, in_shapes, input_rois, border_type, fill_values,
              any_has_degenerated_extents, [&](auto&& roi, auto&& loader) {
                filter::filter2d<<<grid, block, max_total_workspace, ctx.gpu.stream>>>(descs_dev,
                                                                                       roi, loader);
                CUDA_CALL(cudaGetLastError());
              });
  }

 protected:
  template <typename InShapes, typename OutShapes, typename KernelLauncher>
  void RunKernel(KernelContext& ctx, const OutShapes& out_shapes, const InShapes& in_shapes,
                 const span<const filter::InputROI<axes>> input_rois,
                 boundary::BoundaryType border_type,
                 const TensorListView<StorageGPU, const In, 0>& fill_values,
                 bool has_degenerated_extents, KernelLauncher&& launch_kernel) {
    if (input_rois.size() == 0) {
      filter::InputRoiFull<axes> roi_handler{};
      RunKernelWithBorderMode(
          ctx, border_type, fill_values, has_degenerated_extents,
          [&](auto&& loader) { launch_kernel(std::move(roi_handler), std::move(loader)); });
    } else {
      for (int sample_idx = 0; sample_idx < in_shapes.num_samples(); sample_idx++) {
        const auto& out_shape = out_shapes[sample_idx];
        const auto& in_shape = in_shapes[sample_idx];
        ValidateROI(out_shape, in_shape, samples_desc_[sample_idx].shape, input_rois[sample_idx]);
      }
      filter::InputROI<axes>* input_rois_dev;
      std::tie(input_rois_dev) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, input_rois);
      filter::CustomInputROI<axes> roi_handler{input_rois_dev};
      RunKernelWithBorderMode(
          ctx, border_type, fill_values, has_degenerated_extents,
          [&](auto&& loader) { launch_kernel(std::move(roi_handler), std::move(loader)); });
    }
  }

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
        RunKernelBorderRemap<filter::Reflect1001>(ctx, std::move(launch_kernel));
        break;
      case BoundaryType::CLAMP:
        RunKernelBorderRemap<filter::Clamp>(ctx, std::move(launch_kernel));
        break;
      case BoundaryType::WRAP:
        RunKernelBorderRemap<filter::Wrap>(ctx, std::move(launch_kernel));
        break;
      case BoundaryType::CONSTANT:
        RunKernelBorderConstant(ctx, fill_values, std::move(launch_kernel));
        break;
      default:
        DALI_FAIL(
            make_string("Unsupported border type was specified: ", to_string(border_type), "."));
    }
  }

  template <typename Remap, typename KernelLauncher>
  void RunKernelBorderRemap(KernelContext& ctx, KernelLauncher&& launch_kernel) {
    using Loader = filter::InLoaderBorderRemap<Remap, In, axes>;
    filter::InLoaderFactory<Loader> loader_factory{Loader{}};
    launch_kernel(std::move(loader_factory));
  }

  template <typename KernelLauncher>
  void RunKernelBorder101(KernelContext& ctx, bool has_degenerated_extents,
                          KernelLauncher&& launch_kernel) {
    // If any of the samples has some extent equal to 1, border handler needs extra
    // check to prevent infinite loop. Extra check for every single position of the filter
    // over an image is costly, so try to avoid it.
    BOOL_SWITCH(
        has_degenerated_extents, HasDegeneratedExtents,
        (using Loader =
             filter::InLoaderBorderRemap<filter::Reflect101<HasDegeneratedExtents>, In, axes>;
         filter::InLoaderFactory<Loader> loader_factory{Loader{}};
         launch_kernel(std::move(loader_factory));));  // NOLINT
  }

  template <typename KernelLauncher>
  void RunKernelBorderConstant(KernelContext& ctx,
                               const TensorListView<StorageGPU, const In, 0>& fill_values,
                               KernelLauncher&& launch_kernel) {
    int num_samples = samples_desc_.size();
    if (fill_values.num_samples() != num_samples) {
      filter::InLoaderFactory<filter::InLoaderPad<In, axes>> loader_factory{nullptr};
      launch_kernel(std::move(loader_factory));
    } else {
      fill_values_.resize(num_samples);
      for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
        fill_values_[sample_idx] = fill_values[sample_idx].data;
      }
      const In** fill_values_dev;
      std::tie(fill_values_dev) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, fill_values_);
      filter::InLoaderFactory<filter::InLoaderPad<In, axes>> loader_factory{fill_values_dev};
      launch_kernel(std::move(loader_factory));
    }
  }

  template <typename InShape, typename FilterShape>
  filter::ShapeDesc<axes> SetupSampleShapeDesc(int& required_workspace,
                                               bool& has_degenerated_extents, int sample_idx,
                                               const InShape& in_shape,
                                               const FilterShape& filter_shape, ivec<axes> anchor,
                                               int shared_mem_limit) {
    auto r = filter_shape[0];
    auto s = filter_shape[1];
    ivec<axes> filter_extents{r, s};
    auto f = has_sequence_dim ? in_shape[0] : 1;
    auto h = in_shape[num_sequence_dim];
    auto w = in_shape[num_sequence_dim + 1];
    auto c = has_channel_dim ? in_shape[num_sequence_dim + 2] : 1;
    auto wc = w * c;
    auto hwc = h * wc;
    has_degenerated_extents = h == 1 || w == 1;
    ValidateSampleNumericLimits(sample_idx, r, s, volume(filter_shape), anchor[0], anchor[1], f, h,
                                wc, c);
    anchor[1] *= c;
    int max_block_width_log2 = dali::ilog2(block_width);
    int sample_wc_log2 = dali::ilog2(wc);
    int block_width_log2 = std::min(max_block_width_log2, sample_wc_log2);
    auto log_block =
        filter::create_log_block({block_width_log2, max_block_width_log2 - block_width_log2});
    auto lblockDim = log_block.lBlockDim();
    auto in_workspace_width = lblockDim.x + (s - 1) * c;
    auto in_workspace_height = lblockDim.y * lanes + r - 1;
    auto in_workspace_num_elements = in_workspace_width * in_workspace_height;
    if (in_workspace_width > std::numeric_limits<int>::max() ||
        in_workspace_num_elements > std::numeric_limits<int>::max()) {
      in_workspace_width = in_workspace_num_elements = 0;
    }
    required_workspace = in_workspace_num_elements * sizeof(In);
    if (c > lblockDim.x || required_workspace > shared_mem_limit) {
      required_workspace = in_workspace_width = 0;
    }
    return {hwc,
            static_cast<int>(wc),
            static_cast<int>(f),
            static_cast<int>(h),
            static_cast<int>(w),
            static_cast<int>(c),
            filter_extents,
            anchor,
            ivec<axes>{in_workspace_height, in_workspace_width},
            log_block};
  }

  template <typename OutShape>
  void ValidateROI(const OutShape& out_shape, const OutShape& in_shape,
                   const filter::ShapeDesc<axes>& shape_desc, const filter::InputROI<axes>& roi) {
    auto roi_size = roi.end - roi.begin;
    ivec2 out_size{out_shape[num_sequence_dim], out_shape[num_sequence_dim + 1]};
    ivec2 in_size{in_shape[num_sequence_dim], in_shape[num_sequence_dim + 1]};
    DALI_ENFORCE(roi_size == out_size,
                 make_string("The output size must match the input roi size. Got output of size: ",
                             out_size, " and roi of size ", roi_size, "."));
    DALI_ENFORCE(
        all_coords(0 <= roi.begin) && all_coords(roi.end <= in_size),
        make_string("ROI must lie within the input sample. Got roi that starts at: ", roi.begin,
                    " and ends at ", roi.end, " for a sample of shape ", in_size, "."));
  }

  void ValidateSampleNumericLimits(int sample_idx, int64_t r, int64_t s, int64_t filter_vol,
                                   int64_t filter_top_anchor, int64_t filter_left_anchor, int64_t f,
                                   int64_t h, int64_t wc, int64_t c) {
    DALI_ENFORCE(
        filter_vol <= std::numeric_limits<int>::max(),
        make_string("Volume of filter for sample of idx ", sample_idx, " exceeds the limit of ",
                    std::numeric_limits<int>::max(), ". Got: ", filter_vol, "."));
    DALI_ENFORCE(
        f <= std::numeric_limits<int>::max(),
        make_string("Number of frames for sample of idx ", sample_idx, " exceeds the limit of ",
                    std::numeric_limits<int>::max(), ". Got: ", f, "."));
    DALI_ENFORCE(h <= max_sample_height,
                 make_string("The height of sample of idx ", sample_idx, " exceeds the limit of ",
                             max_sample_height, ". Got: ", h, "."));
    DALI_ENFORCE(0 <= wc && wc <= max_sample_width,
                 make_string("The total width and number of channels in sample of idx ", sample_idx,
                             " exceeds the limit of ", max_sample_width, ". Got: ", wc, "."));
    auto height_radious = h + r - filter_top_anchor - 2;
    DALI_ENFORCE(
        0 <= height_radious && height_radious <= std::numeric_limits<int>::max(),
        make_string("The combined height of the sample and filter radious for sample of idx ",
                    sample_idx, " exceeds the limit of ", std::numeric_limits<int>::max(),
                    ". Got: ", height_radious, "."));
    auto width_radious = wc - 1 + (s - 1 - filter_left_anchor) * c;
    DALI_ENFORCE(
        0 <= width_radious && width_radious <= std::numeric_limits<int>::max(),
        make_string("The combined width, number of channels and filter radious for sample of idx ",
                    sample_idx, " exceeds the limit of ", std::numeric_limits<int>::max(),
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
