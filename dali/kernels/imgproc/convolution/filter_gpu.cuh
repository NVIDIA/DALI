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

template <int N, typename T>
vec<N, T> rev(const vec<N, T>& v) {
  vec<N, T> out;
  for (int d = 0; d < N; d++) {
    out[N - d - 1] = v[d];
  }
  return out;
}

// Descrption of the input ROI passed to the kernel
template <int axes>
struct InputROI {
  ivec<axes> start, end;
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
  int64_t frame_stride;
  i64vec<axes> in_strides;
  int num_frames, width, num_channels;
  ivec<axes> in_extents;
  ivec<axes> filter_extents;
  ivec<axes> anchor_shift;
  ivec<axes> workspace_extents;
  ivec<axes> workspace_strides;
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

template <int axes>
struct InputRoiFull {
  struct ROI {
    DALI_HOST_DEV ivec<axes> start() const {
      return 0;
    }

    DALI_HOST_DEV ivec<axes> size() const {
      return shape_desc_.in_extents;
    }

    DALI_HOST_DEV int64_t frame_stride() const {
      return shape_desc_.frame_stride;
    }

    DALI_HOST_DEV i64vec<axes> get_strides() const {
      return shape_desc_.in_strides;
    }

    const ShapeDesc<axes>& shape_desc_;
  };

  DALI_DEVICE DALI_FORCEINLINE ROI operator()(const ShapeDesc<axes>& shape_desc, int sample_idx) {
    (void)sample_idx;
    return {shape_desc};
  }
};

template <int axes>
struct CustomInputROI {
  struct ROI {
    DALI_HOST_DEV ivec<axes> start() const {
      return start_;
    }

    DALI_HOST_DEV ivec<axes> size() const {
      return size_;
    }

    DALI_HOST_DEV int64_t frame_stride() const {
      return frame_stride_;
    }

    // todo make it a member?
    template <int axes_ = axes>
    DALI_HOST_DEV std::enable_if_t<axes_ == 2, i64vec2> get_strides() const {
      return {1, size_[0]};
    }

    ivec<axes> start_, size_;
    int64_t frame_stride_;
  };

  DALI_DEVICE DALI_FORCEINLINE ROI operator()(const ShapeDesc<axes>& shape_desc, int sample_idx) {
    auto roi = rois[sample_idx];
    roi.start[0] *= shape_desc.num_channels;
    roi.end[0] *= shape_desc.num_channels;
    auto size = roi.end - roi.start;
    return {roi.start, size, volume(size)};
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
  using Remap::border_remap;

  DALI_HOST_DEV DALI_FORCEINLINE int border_remap(int idx, const ShapeDesc<axes>& sample_shape,
                                                  int axis) const {
    return border_remap(idx, sample_shape.in_extents[axis]);
  }

  DALI_HOST_DEV DALI_FORCEINLINE int border_remap_innermost(
      int idx, const ShapeDesc<axes>& sample_shape) const {
    int num_channels = sample_shape.num_channels;
    if (idx < 0) {
      int reflect_dim_idx = (idx + 1) / num_channels - 1;
      int inner_dim_idx = (idx + 1) % num_channels + num_channels - 1;
      return border_remap(reflect_dim_idx, sample_shape.width) * num_channels + inner_dim_idx;
    }
    if (idx >= sample_shape.in_extents[0]) {
      return border_remap(idx / num_channels, sample_shape.width) * num_channels +
             idx % num_channels;
    }
    return idx;
  }

  DALI_HOST_DEV DALI_FORCEINLINE In load(const In* __restrict__ in, const ivec<axes>& coords,
                                         const ShapeDesc<axes>& sample_shape) const {
    return in[dot(coords, sample_shape.in_strides)];
  }
};

template <typename In, int axes>
struct InLoaderPad {
  DALI_HOST_DEV DALI_FORCEINLINE int border_remap(int idx, const ShapeDesc<axes>& sample_shape,
                                                  int axis) const {
    (void)sample_shape;
    (void)axis;
    return idx;
  }

  DALI_HOST_DEV DALI_FORCEINLINE int border_remap_innermost(
      int idx, const ShapeDesc<axes>& sample_shape) const {
    (void)sample_shape;
    return idx;
  }

  DALI_HOST_DEV DALI_FORCEINLINE In load(const In* __restrict__ in, const ivec<axes>& coords,
                                         const ShapeDesc<axes>& sample_shape) const {
    if (any_coord(coords < 0) || any_coord(coords >= sample_shape.in_extents)) {
      return fill_value;
    }
    return in[dot(coords, sample_shape.in_strides)];
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
                                            const ivec2& start) const {
    const auto& log_block = sample_desc.shape.log_block;
    const auto& lBlockDim = log_block.lBlockDim();
    const auto& lThreadIdx = log_block.lThreadIdx();
    __syncthreads();
    load_input_to_shm(in, start);
    __syncthreads();
    const auto* filter = sample_desc.filter;
    for (int r = 0; r < sample_desc.shape.filter_extents[1]; r++) {
      for (int s = 0; s < sample_desc.shape.filter_extents[0]; s++) {
        auto filter_coef = __ldg(filter++);
#pragma unroll
        for (int lane = 0, lanes_offset = 0; lane < SampleDescT::lanes;
             lane++, lanes_offset += lBlockDim.y) {
          ivec2 filter_offset{s * sample_desc.shape.num_channels, r + lanes_offset};
          auto in_val =
              in_workspace[dot(lThreadIdx + filter_offset, sample_desc.shape.workspace_strides)];
          acc[lane] += in_val * filter_coef;
        }
      }
    }
  }

  DALI_DEVICE DALI_FORCEINLINE void load_input_to_shm(const In* __restrict__ in,
                                                      const ivec2& start) const {
    const auto& log_block = sample_desc.shape.log_block;
    const auto& lBlockDim = log_block.lBlockDim();
    const auto& lThreadIdx = log_block.lThreadIdx();
    auto start_shifted = start - sample_desc.shape.anchor_shift;
    for (int x = lThreadIdx.x; x < sample_desc.shape.workspace_extents[0]; x += lBlockDim.x) {
      int global_x = in_loader.border_remap_innermost(start_shifted[0] + x, sample_desc.shape);
#pragma unroll SampleDescT::lanes
      for (int y = lThreadIdx.y; y < sample_desc.shape.workspace_extents[1]; y += lBlockDim.y) {
        int global_y = in_loader.border_remap(start_shifted[1] + y, sample_desc.shape, 1);
        in_workspace[dot(ivec2{x, y}, sample_desc.shape.workspace_strides)] =
            in_loader.load(in, ivec2{global_x, global_y}, sample_desc.shape);
      }
    }
  }

  const SampleDescT& sample_desc;
  const Inloader& in_loader;
  In* in_workspace;
};

template <typename SampleDescT, typename Inloader, typename In>
DALI_DEVICE DALI_FORCEINLINE ShmInputConv<SampleDescT, Inloader> create_shm_conv(
    const SampleDescT& sample_desc, const Inloader& in_loader, In* in_workspace) {
  return {sample_desc, in_loader, in_workspace};
}

/**
 * @brief Computes the convolution of size ``block_width x lanes`` accessing the input directly in
 * global memory. Used as a fallback when the filter size of number of channels in the input makes
 * it impossible to use ``ShmInputConv``.
 */
template <typename SampleDescT, typename Inloader>
struct DirectInputConv {
  using Acc = typename SampleDescT::Acc;
  using In = typename SampleDescT::In;

  DALI_DEVICE DALI_FORCEINLINE void compute(Acc* __restrict__ acc, const In* __restrict__ in,
                                            const ivec2& start) const {
    const auto& log_block = sample_desc.shape.log_block;
    const auto& lBlockDim = log_block.lBlockDim();
    const auto& lThreadIdx = log_block.lThreadIdx();
    auto start_shifted = start - sample_desc.shape.anchor_shift;
    const auto* filter = sample_desc.filter;
    for (int s = 0; s < sample_desc.shape.filter_extents[0]; s++) {
      auto global_x = in_loader.border_remap_innermost(
          start_shifted[0] + lThreadIdx.x + s * sample_desc.shape.num_channels, sample_desc.shape);
      for (int r = 0; r < sample_desc.shape.filter_extents[1]; r++) {
        auto filter_coef = __ldg(filter + r * sample_desc.shape.filter_extents[0] + s);
        // Even without shm, using `lanes` speeds up the kernel by reducing
        // the cost of nested loops arithmetic per single output value
#pragma unroll
        for (int lane = 0; lane < SampleDescT::lanes; lane++) {
          auto global_y = in_loader.border_remap(
              start_shifted[1] + lThreadIdx.y + lane * lBlockDim.y + r, sample_desc.shape, 1);
          auto in_val = in_loader.load(in, ivec2{global_x, global_y}, sample_desc.shape);
          acc[lane] += in_val * filter_coef;
        }
      }
    }
  }

  const SampleDescT& sample_desc;
  const Inloader& in_loader;
};


template <typename SampleDescT, typename Inloader>
DALI_DEVICE DALI_FORCEINLINE DirectInputConv<SampleDescT, Inloader> create_direct_conv(
    const SampleDescT& sample_desc, const Inloader& in_loader) {
  return {sample_desc, in_loader};
}


/** @} */  // end of InputConv

template <int lanes, typename Out, typename Acc, typename ROI, int axes>
DALI_DEVICE DALI_FORCEINLINE void store_acc_in_global_output(Out* __restrict__ out,
                                                             const Acc* __restrict__ acc,
                                                             const ROI& roi,
                                                             const ivec<axes>& start,
                                                             const LogicalBlock<axes>& log_block) {
  const auto& lBlockDim = log_block.lBlockDim();
  const auto& lThreadIdx = log_block.lThreadIdx();
  const auto& roi_size = roi.size();
  const auto& roi_strides = roi.get_strides();
  int x = start[0] + lThreadIdx.x;
  if (x < roi_size[0]) {
#pragma unroll
    for (int lane = 0; lane < lanes; lane++) {
      int y = start[1] + lThreadIdx.y + lane * lBlockDim.y;
      if (y < roi_size[1]) {
        out[dot(ivec2{x, y}, roi_strides)] = ConvertSat<Out>(acc[lane]);
      }
    }
  }
}

template <typename SampleDescT, typename Conv, typename ROI>
DALI_DEVICE DALI_FORCEINLINE void stride_grid(const ivec2& block_start,
                                              const SampleDescT& sample_desc, const ROI& roi,
                                              const Conv& conv) {
  constexpr int lanes = SampleDescT::lanes;
  const auto* in = sample_desc.in;
  auto* out = sample_desc.out;
  const auto& log_block = sample_desc.shape.log_block;
  const auto& lBlockDim = log_block.lBlockDim();
  const auto& roi_size = roi.size();
  for (int f = 0; f < sample_desc.shape.num_frames;
       f++, in += sample_desc.shape.frame_stride, out += roi.frame_stride()) {
    for (int y_start = block_start.y; y_start < roi_size[1];
         y_start += lanes * lBlockDim.y * gridDim.y) {
      for (int x_start = block_start.x; x_start < roi_size[0]; x_start += gridDim.x * lBlockDim.x) {
        typename SampleDescT::Acc acc[lanes] = {};
        ivec2 start{x_start, y_start};
        conv.compute(acc, in, start + roi.start());
        store_acc_in_global_output<lanes>(out, acc, roi, start, log_block);
      }
    }
  }
}

template <typename SampleDescT>
DALI_DEVICE DALI_FORCEINLINE std::enable_if_t<SampleDescT::axes == 2, ivec2> get_start_block(
    const SampleDescT& sample_desc) {
  constexpr int lanes = SampleDescT::lanes;
  const auto& lBlockDim = sample_desc.shape.log_block.lBlockDim();
  return {lBlockDim.x * blockIdx.x, lanes * lBlockDim.y * blockIdx.y};
}

template <typename SampleDescT, typename InputROIFactory, typename InLoaderFactory>
__global__ void filter(const SampleDescT* __restrict__ descs, InputROIFactory in_roi_factory,
                       InLoaderFactory in_loader_factory) {
  extern __shared__ char shm[];
  auto sample_desc = descs[blockIdx.z];
  auto&& roi = in_roi_factory(sample_desc.shape, blockIdx.z);
  auto block_start = get_start_block(sample_desc);
  if (any_coord(block_start >= roi.size())) {
    return;
  }
  auto&& in_loader = in_loader_factory(blockIdx.z);
  if (sample_desc.shape.workspace_extents[0]) {
    using In = typename SampleDescT::In;
    In* in_workspace = reinterpret_cast<In*>(shm);
    auto conv = create_shm_conv(sample_desc, in_loader, in_workspace);
    stride_grid(block_start, sample_desc, roi, conv);
  } else {
    auto conv = create_direct_conv(sample_desc, in_loader);
    stride_grid(block_start, sample_desc, roi, conv);
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
                filter::filter<<<grid, block, max_total_workspace, ctx.gpu.stream>>>(descs_dev, roi,
                                                                                     loader);
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
    ivec<axes> filter_extents{s, r};
    auto f = has_sequence_dim ? in_shape[0] : 1;
    auto h = in_shape[num_sequence_dim];
    auto w = in_shape[num_sequence_dim + 1];
    auto c = has_channel_dim ? in_shape[num_sequence_dim + 2] : 1;
    auto wc = w * c;
    auto frame_stride = h * wc;
    has_degenerated_extents = h == 1 || w == 1;
    ValidateSampleNumericLimits(sample_idx, r, s, volume(filter_shape), anchor[1], anchor[0], f, h,
                                wc, c);
    anchor[0] *= c;
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
    return {frame_stride,
            {1, wc},
            static_cast<int>(f),
            static_cast<int>(w),
            static_cast<int>(c),
            {wc, h},
            filter_extents,
            anchor,
            ivec<axes>{in_workspace_width, in_workspace_height},
            {1, in_workspace_width},
            log_block};
  }

  template <typename OutShape>
  void ValidateROI(const OutShape& out_shape, const OutShape& in_shape,
                   const filter::ShapeDesc<axes>& shape_desc, const filter::InputROI<axes>& roi) {
    auto roi_size = roi.end - roi.start;
    ivec2 out_size{out_shape[num_sequence_dim + 1], out_shape[num_sequence_dim]};
    ivec2 in_size{in_shape[num_sequence_dim + 1], in_shape[num_sequence_dim]};
    DALI_ENFORCE(
        roi_size == out_size,
        make_string("The output size must match the input roi size. Got output of size: ",
                    filter::rev(out_size), " and roi of size ", filter::rev(roi_size), "."));
    DALI_ENFORCE(all_coords(0 <= roi.start) && all_coords(roi.end <= in_size),
                 make_string("ROI must lie within the input sample. Got roi that starts at: ",
                             roi.start, " and ends at ", filter::rev(roi.end),
                             " for a sample of shape ", filter::rev(in_size), "."));
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
