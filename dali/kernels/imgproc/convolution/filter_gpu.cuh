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
  ivec<axes> start, end;
};

template <int N, typename T>
vec<N, T> rev(const vec<N, T>& v) {
  vec<N, T> out;
  for (int d = 0; d < N; d++) {
    out[N - d - 1] = v[d];
  }
  return out;
}

template <int axes_, int lanes_, int block_size>
struct StaticBlock {
  struct LogicalBlock {
    static constexpr int axes = axes_;
    static constexpr int lanes = lanes_;

    DALI_HOST_DEV DALI_FORCEINLINE int flat_size() const {
      return block_size;
    }

    DALI_HOST_DEV DALI_FORCEINLINE ivec<axes> lBlockDim() const {
      return cat(ivec<1>{block_size}, ivec<axes - 1>{1});
    }

    DALI_DEVICE DALI_FORCEINLINE int flat_idx() const {
      return threadIdx.x;
    }

    DALI_DEVICE DALI_FORCEINLINE ivec<axes> lThreadIdx() const {
      return cat(ivec<1>{threadIdx.x}, ivec<axes - 1>{0});
    }
  };

  DALI_DEVICE DALI_FORCEINLINE LogicalBlock operator()(int sample_idx) {
    (void)sample_idx;
    return {};
  }
};

template <int axes_, int lanes_>
struct AdaptiveBlock {
  struct LogicalBlock {
    static constexpr int axes = axes_;
    static constexpr int lanes = lanes_;

    ivec<axes> extents_log2;
    ivec<axes> strides_log2;

    DALI_DEVICE DALI_FORCEINLINE int flat_size() const {
      assert(blockDim.y == 1);
      assert(blockDim.z == 1);
      return blockDim.x;
    }

    DALI_HOST_DEV DALI_FORCEINLINE ivec<axes> lBlockDim() const {
      return 1 << extents_log2;
    }

    DALI_DEVICE DALI_FORCEINLINE int flat_idx() const {
      return threadIdx.x;
    }

    DALI_DEVICE DALI_FORCEINLINE ivec<axes> lThreadIdx() const {
      assert(threadIdx.y == 0);
      assert(threadIdx.z == 0);
      return (int(threadIdx.x) >> strides_log2) & (lBlockDim() - 1);
    }
  };

  DALI_DEVICE DALI_FORCEINLINE const LogicalBlock& operator()(int sample_idx) {
    return blocks[sample_idx];
  }

  const LogicalBlock* blocks;
};

template <int lanes>
typename AdaptiveBlock<2, lanes>::LogicalBlock create_adaptive_block(ivec2 xy_log2) {
  return {xy_log2, {0, xy_log2.x}};
}

template <int axes>
struct LogGrid {};

template <>
struct LogGrid<2> {
  DALI_HOST_DEV ivec<2> lGridDim() const {
    return num_blocks_;
  }

  DALI_DEVICE ivec<2> lBlockIdx() const {
    return {blockIdx.x, blockIdx.y};
  }

  DALI_HOST_DEV int num_samples() const {
    return num_samples_;
  }

  DALI_DEVICE int sample_idx() const {
    return blockIdx.z;
  }

  template <typename LogBlockT>
  DALI_DEVICE ivec2 block_position(const LogBlockT& log_block) const {
    static_assert(LogBlockT::axes == 2);
    auto block_dim = log_block.lBlockDim();
    auto block_idx = lBlockIdx();
    return {block_dim.x * block_idx.x, LogBlockT::lanes * block_dim.y * block_idx.y};
  }

  template <typename LogBlockT>
  DALI_HOST_DEV ivec2 grid_size(const LogBlockT& log_block) const {
    static_assert(LogBlockT::axes == 2);
    auto grid_dim = lGridDim();
    auto block_dim = log_block.lBlockDim();
    return {grid_dim.x * block_dim.x, LogBlockT::lanes * block_dim.y * grid_dim.y};
  }

  dim3 gird_setup() const {
    auto grid = lGridDim();
    return {static_cast<unsigned int>(grid.x), static_cast<unsigned int>(grid.y),
            static_cast<unsigned int>(num_samples())};
  }

  ivec2 num_blocks_;
  int num_samples_;
};

inline LogGrid<2> create_log_grid(ivec2 num_blocks, int num_samples) {
  return {num_blocks, num_samples};
}


template <int axes>
struct ShapeDesc {
  int64_t frame_stride;
  i64vec<axes> in_strides;
  int num_frames, width, num_channels;
  ivec<axes> in_extents;
  ivec<axes> filter_extents;
  ivec<axes> anchor_shift;
  ivec<axes> in_workspace_extents;
  ivec<axes> in_workspace_strides;
  int in_workspace_offset;
};

template <typename Out_, typename In_, typename W_, typename Acc_, int axes_>
struct SampleDesc {
  using Acc = Acc_;
  using Out = Out_;
  using In = In_;
  using W = W_;
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

    DALI_HOST_DEV i64vec<axes> get_strides() const {
      return strides_;
    }

    ivec<axes> start_, size_;
    i64vec<axes> strides_;
    int64_t frame_stride_;
  };

  DALI_DEVICE DALI_FORCEINLINE const ROI& operator()(const ShapeDesc<axes>& shape_desc,
                                                     int sample_idx) {
    (void)shape_desc;
    return rois[sample_idx];
  }

  const ROI* rois;
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
  DALI_DEVICE DALI_FORCEINLINE InLoader operator()(int sample_idx) {
    return {};
  }
};

template <typename In, int axes>
struct InLoaderFactory<InLoaderPad<In, axes>> {
  DALI_DEVICE DALI_FORCEINLINE InLoaderPad<In, axes> operator()(int sample_idx) {
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
template <typename SampleDescT_, typename Inloader_, typename LogBlockT_>
struct ShmInputConv {
  using SampleDescT = SampleDescT_;
  using Inloader = Inloader_;
  using LogBlockT = LogBlockT_;
  using In = typename SampleDescT::In;
  using Acc = typename SampleDescT::Acc;

  DALI_DEVICE DALI_FORCEINLINE void compute(Acc* __restrict__ acc, const In* __restrict__ in,
                                            const ivec2& start) const {
    auto anchored_start = start - sample_desc.shape.anchor_shift;
    __syncthreads();
    precompute_indices(in, anchored_start);
    __syncthreads();
    load_input_to_shm(in, anchored_start);
    __syncthreads();
    const auto& lThreadIdx = log_block.lThreadIdx();
    stride_filter(sample_desc.shape.filter_extents, [&](auto filter_coef, const auto& filter_offset,
                                                        int lane) {
      auto in_val =
          in_workspace[dot(lThreadIdx + filter_offset, sample_desc.shape.in_workspace_strides)];
      acc[lane] += in_val * filter_coef;
    });
  }

  template <typename MulAddCoef>
  DALI_DEVICE DALI_FORCEINLINE void stride_filter(const ivec2& filter_extents,
                                                  MulAddCoef&& mul_add_coef) const {
    const auto& lBlockDim = log_block.lBlockDim();
    const auto* filter = sample_desc.filter;
    for (int r = 0; r < filter_extents[1]; r++) {
      for (int s = 0; s < filter_extents[0]; s++) {
        auto filter_coef = __ldg(filter++);
#pragma unroll
        for (int lane = 0, lanes_offset = 0; lane < LogBlockT::lanes;
             lane++, lanes_offset += lBlockDim.y) {
          ivec2 filter_offset{s * sample_desc.shape.num_channels, r + lanes_offset};
          mul_add_coef(filter_coef, filter_offset, lane);
        }
      }
    }
  }

  DALI_DEVICE DALI_FORCEINLINE void precompute_indices(const In* __restrict__ in,
                                                       const ivec2& anchored_start) const {
    for (int y = log_block.flat_idx(); y < sample_desc.shape.in_workspace_extents.y;
         y += log_block.flat_size()) {
      idx_cache[y] = in_loader.border_remap(anchored_start[1] + y, sample_desc.shape, 1);
    }
  }

  DALI_DEVICE DALI_FORCEINLINE void load_input_to_shm(const In* __restrict__ in,
                                                      const ivec2& anchored_start) const {
    const auto& lBlockDim = log_block.lBlockDim();
    const auto& lThreadIdx = log_block.lThreadIdx();
    for (int x = lThreadIdx.x; x < sample_desc.shape.in_workspace_extents[0]; x += lBlockDim.x) {
      int global_x = in_loader.border_remap_innermost(anchored_start[0] + x, sample_desc.shape);
      for (int y = lThreadIdx.y; y < sample_desc.shape.in_workspace_extents[1]; y += lBlockDim.y) {
        int global_y = idx_cache[y];
        in_workspace[dot(ivec2{x, y}, sample_desc.shape.in_workspace_strides)] =
            in_loader.load(in, ivec2{global_x, global_y}, sample_desc.shape);
      }
    }
  }

  const SampleDescT& sample_desc;
  const Inloader& in_loader;
  const LogBlockT& log_block;
  In* in_workspace;
  int* idx_cache;
};

template <typename SampleDescT, typename Inloader, typename LogBlockT, typename In>
DALI_DEVICE DALI_FORCEINLINE ShmInputConv<SampleDescT, Inloader, LogBlockT> create_shm_conv(
    const SampleDescT& sample_desc, const Inloader& in_loader, const LogBlockT& log_block,
    In* in_workspace, int* idx_cache) {
  return {sample_desc, in_loader, log_block, in_workspace, idx_cache};
}

/**
 * @brief Computes the convolution of size ``block_width x lanes`` accessing the input directly in
 * global memory. Used as a fallback when the filter size of number of channels in the input makes
 * it impossible to use ``ShmInputConv``.
 */
template <typename SampleDescT_, typename Inloader_, typename LogBlockT_>
struct DirectInputConv {
  using SampleDescT = SampleDescT_;
  using Inloader = Inloader_;
  using LogBlockT = LogBlockT_;
  using Acc = typename SampleDescT::Acc;
  using In = typename SampleDescT::In;

  DALI_DEVICE DALI_FORCEINLINE void compute(Acc* __restrict__ acc, const In* __restrict__ in,
                                            const ivec2& start) const {
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
        for (int lane = 0; lane < LogBlockT::lanes; lane++) {
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
  const LogBlockT& log_block;
};


template <typename SampleDescT, typename Inloader, typename LogBlockT>
DALI_DEVICE DALI_FORCEINLINE DirectInputConv<SampleDescT, Inloader, LogBlockT> create_direct_conv(
    const SampleDescT& sample_desc, const Inloader& in_loader, const LogBlockT& log_block) {
  return {sample_desc, in_loader, log_block};
}


/** @} */  // end of InputConv

template <typename LogBlockT, typename Cb>
DALI_DEVICE DALI_FORCEINLINE void for_each_output_point_in_block(const ivec2& block_start,
                                                                 const ivec2& roi_size,
                                                                 const LogBlockT& log_block,
                                                                 const Cb&& cb) {
  const auto& lBlockDim = log_block.lBlockDim();
  auto coords = block_start + log_block.lThreadIdx();
  if (coords.x < roi_size.x) {
#pragma unroll
    for (int lane = 0; lane < LogBlockT::lanes; lane++) {
      if (coords.y < roi_size.y) {
        cb(coords, lane);
      }
      coords.y += lBlockDim.y;
    }
  }
}

template <typename ROI, typename DoConv, typename In, typename Out>
DALI_DEVICE DALI_FORCEINLINE void stride_grid(const ivec2& block_start, const ivec2& grid_extents,
                                              const ROI& roi, const In* __restrict__ in,
                                              Out* __restrict__ out, DoConv&& do_conv) {
  const auto& roi_size = roi.size();
  for (int y_start = block_start.y; y_start < roi_size[1]; y_start += grid_extents.y) {
    for (int x_start = block_start.x; x_start < roi_size[0]; x_start += grid_extents.x) {
      do_conv(ivec2{x_start, y_start}, in, out);
    }
  }
}

template <typename ROI, typename Conv, int axes>
DALI_DEVICE DALI_FORCEINLINE void stride_grid(const ivec<axes>& initial_block_start,
                                              const ivec<axes>& grid_extents, const ROI& roi,
                                              const Conv& conv) {
  using LogBlockT = typename Conv::LogBlockT;
  using SampleDescT = typename Conv::SampleDescT;
  using Acc = typename SampleDescT::Acc;
  using Out = typename SampleDescT::Out;
  constexpr int lanes = LogBlockT::lanes;
  const auto* in = conv.sample_desc.in;
  auto* out = conv.sample_desc.out;
  const auto roi_strides = roi.get_strides();
  for (int f = 0; f < conv.sample_desc.shape.num_frames;
       f++, in += conv.sample_desc.shape.frame_stride, out += roi.frame_stride()) {
    stride_grid(
        initial_block_start, grid_extents, roi, in, out,
        [&](const ivec<axes>& block_start, const auto* __restrict__ in, auto* __restrict__ out) {
          Acc acc[lanes]{};
          conv.compute(acc, in, block_start + roi.start());
          for_each_output_point_in_block(
              block_start, roi.size(), conv.log_block, [&](const auto& coords, int lane) {
                out[dot(coords, roi_strides)] = ConvertSat<Out>(acc[lane]);
              });
        });
  }
}

template <typename SampleDescT, typename InputROIFactory, typename LogicalBlockFactory,
          typename InLoaderFactory, typename LogGridT>
__global__ void filter(const SampleDescT* __restrict__ descs, InputROIFactory in_roi_factory,
                       LogicalBlockFactory log_block_factory, InLoaderFactory in_loader_factory,
                       LogGridT log_grid) {
  extern __shared__ char shm[];
  int sample_idx = log_grid.sample_idx();
  auto sample_desc = descs[sample_idx];
  const auto& roi = in_roi_factory(sample_desc.shape, sample_idx);
  const auto& log_block = log_block_factory(sample_idx);
  auto block_start = log_grid.block_position(log_block);
  if (any_coord(block_start >= roi.size())) {
    return;  // early exit to avoid all the setup only to do nothing in the stride loop
  }
  auto grid_size = log_grid.grid_size(log_block);
  const auto& in_loader = in_loader_factory(sample_idx);
  if (sample_desc.shape.in_workspace_extents[0]) {
    using In = typename SampleDescT::In;
    int* idx_cache = reinterpret_cast<int*>(shm);
    In* in_workspace = reinterpret_cast<In*>(shm + sample_desc.shape.in_workspace_offset);
    auto conv = create_shm_conv(sample_desc, in_loader, log_block, in_workspace, idx_cache);
    stride_grid(block_start, grid_size, roi, conv);
  } else {
    auto conv = create_direct_conv(sample_desc, in_loader, log_block);
    stride_grid(block_start, grid_size, roi, conv);
  }
}
}  // namespace filter

template <typename Out, typename In, typename W, bool has_channel_dim, bool has_sequence_dim,
          int axes_>
struct FilterGpu {
  /* It computes a corellation of the input and the filter.
  Flip filter in both dimensions for a convolution. */

  static constexpr int axes = axes_;
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

  using SampleDescT = filter::SampleDesc<Out, In, W, Intermediate, axes>;
  using LogBlockFactoryT = filter::AdaptiveBlock<axes, lanes>;
  using LogBlockT = typename LogBlockFactoryT::LogicalBlock;
  using StaticBlockFactoryT = filter::StaticBlock<axes, lanes, block_width>;
  using StaticBlockT = typename StaticBlockFactoryT::LogicalBlock;
  using CustomROIFactoryT = typename filter::CustomInputROI<axes>;
  using CustomROIT = typename CustomROIFactoryT::ROI;
  using LogGridT = filter::LogGrid<axes>;

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
    log_blocks_.clear();
    log_blocks_.reserve(num_samples);
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      const auto& in_shape = in_shapes[sample_idx];
      const auto& filter_shape = filter_shapes[sample_idx];
      int required_workspace;
      bool has_degenerated_extents;
      // todo split validation and log_block creation and sample creation?
      LogBlockT log_block;
      auto shape_desc =
          SetupSampleShapeDesc(required_workspace, has_degenerated_extents, log_block, sample_idx,
                               in_shape, filter_shape, anchors[sample_idx], shared_mem_limit);
      any_has_degenerated_extents |= has_degenerated_extents;
      max_total_workspace = std::max(max_total_workspace, required_workspace);
      log_blocks_.push_back(log_block);
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
    auto log_grid = filter::create_log_grid(ivec2{num_blocks_w, num_blocks_h}, num_samples);
    RunKernel(ctx, out_shapes, in_shapes, input_rois, border_type, fill_values,
              any_has_degenerated_extents, [&](auto&& roi) {
                return [&](auto&& log_block) {
                  return [&](auto&& loader) {
                    filter::filter<<<log_grid.gird_setup(), block_width, max_total_workspace,
                                     ctx.gpu.stream>>>(descs_dev, roi, log_block, loader, log_grid);
                    CUDA_CALL(cudaGetLastError());
                  };
                };
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
      RunKernelWithLogicalBlock(ctx, border_type, fill_values, has_degenerated_extents,
                                launch_kernel(std::move(roi_handler)));
    } else {
      custom_rois_.clear();
      custom_rois_.reserve(in_shapes.num_samples());
      for (int sample_idx = 0; sample_idx < in_shapes.num_samples(); sample_idx++) {
        const auto& out_shape = out_shapes[sample_idx];
        const auto& in_shape = in_shapes[sample_idx];
        custom_rois_.push_back(SetupValidateROI(
            out_shape, in_shape, samples_desc_[sample_idx].shape, input_rois[sample_idx]));
      }
      CustomROIT* input_rois_dev;
      std::tie(input_rois_dev) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, custom_rois_);
      CustomROIFactoryT roi_handler{input_rois_dev};
      RunKernelWithLogicalBlock(ctx, border_type, fill_values, has_degenerated_extents,
                                launch_kernel(std::move(roi_handler)));
    }
  }

  template <typename KernelLauncher>
  void RunKernelWithLogicalBlock(KernelContext& ctx, boundary::BoundaryType border_type,
                                 const TensorListView<StorageGPU, const In, 0>& fill_values,
                                 bool has_degenerated_extents, KernelLauncher&& launch_kernel) {
    if (std::all_of(log_blocks_.begin(), log_blocks_.end(), [](const auto& log_block) {
          return log_block.lBlockDim() == StaticBlockT{}.lBlockDim();
        })) {
      RunKernelWithBorderMode(ctx, border_type, fill_values, has_degenerated_extents,
                              launch_kernel(StaticBlockFactoryT{}));
    } else {
      LogBlockT* log_blocks_dev;
      std::tie(log_blocks_dev) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, log_blocks_);
      LogBlockFactoryT log_block{log_blocks_dev};
      RunKernelWithBorderMode(ctx, border_type, fill_values, has_degenerated_extents,
                              launch_kernel(std::move(log_block)));
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
    filter::InLoaderFactory<Loader> loader_factory{};
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
         filter::InLoaderFactory<Loader> loader_factory{};
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
                                               bool& has_degenerated_extents, LogBlockT& log_block,
                                               int sample_idx, const InShape& in_shape,
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
    log_block = SetupLogicalBlock(ivec2{wc, h});
    auto lblockDim = log_block.lBlockDim();
    auto in_workspace_width = lblockDim.x + (s - 1) * c;
    auto in_workspace_height = lblockDim.y * lanes + r - 1;
    auto in_workspace_num_elements = in_workspace_width * in_workspace_height;
    if (in_workspace_width > std::numeric_limits<int>::max() ||
        in_workspace_num_elements > std::numeric_limits<int>::max()) {
      in_workspace_width = in_workspace_num_elements = 0;
    }
    int idx_workspace_size = in_workspace_height;
    int in_workspace_offset = align_up((idx_workspace_size) * sizeof(int), sizeof(In));
    required_workspace = in_workspace_offset + in_workspace_num_elements * sizeof(In);
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
            in_workspace_offset};
  }

  LogBlockT SetupLogicalBlock(ivec<axes> in_extents) {
    int max_block_width_log2 = dali::ilog2(block_width);
    int sample_wc_log2 = in_extents[0] == 0 ? 0 : dali::ilog2(in_extents[0] - 1) + 1;
    int block_width_log2 = std::min(max_block_width_log2, sample_wc_log2);
    return filter::create_adaptive_block<lanes>(
        {block_width_log2, max_block_width_log2 - block_width_log2});
  }

  template <typename OutShape>
  CustomROIT SetupValidateROI(const OutShape& out_shape, const OutShape& in_shape,
                              const filter::ShapeDesc<axes>& shape_desc,
                              const filter::InputROI<axes>& roi) {
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
    auto roi_start = roi.start;
    roi_start[0] *= shape_desc.num_channels;
    roi_size[0] *= shape_desc.num_channels;
    return {roi_start, roi_size, {1, roi_size[0]}, volume(roi_size)};
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
  std::vector<CustomROIT> custom_rois_;
  std::vector<LogBlockT> log_blocks_;
};


// WAR c++14 odr usage issue (make_string in error message takes them as l-values)
// it should be unnecessary in c++17
template <typename Out, typename In, typename W, bool has_channel_dim, bool has_sequence_dim,
          int axes>
constexpr int FilterGpu<Out, In, W, has_channel_dim, has_sequence_dim, axes>::max_sample_height;

template <typename Out, typename In, typename W, bool has_channel_dim, bool has_sequence_dim,
          int axes>
constexpr int FilterGpu<Out, In, W, has_channel_dim, has_sequence_dim, axes>::max_sample_width;

template <typename Out, typename In, typename W, bool has_channel_dim, bool has_sequence_dim,
          int axes>
constexpr int FilterGpu<Out, In, W, has_channel_dim, has_sequence_dim, axes>::max_grid_height;

template <typename Out, typename In, typename W, bool has_channel_dim, bool has_sequence_dim,
          int axes>
constexpr int FilterGpu<Out, In, W, has_channel_dim, has_sequence_dim, axes>::max_grid_width;

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_FILTER_GPU_H_
