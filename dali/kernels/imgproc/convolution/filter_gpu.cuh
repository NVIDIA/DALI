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
#include "dali/kernels/imgproc/roi.h"
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

template <int N, typename T, typename U>
void strides(vec<N, U>& out, U& total_stride, const vec<N, T>& v) {
  total_stride = 1;
  for (int d = 0; d < N; d++) {
    out[d] = total_stride;
    total_stride *= v[d];
  }
}

template <int N, typename T, typename U>
void strides(vec<N, U>& out, const vec<N, T>& v) {
  U total_strides;
  strides(out, total_strides, v);
}

template <typename StaticConfigT_>
struct StaticBlock {
  struct BlockSetup {
    using StaticConfigT = StaticConfigT_;
    static constexpr int threadblock_size = StaticConfigT::threadblock_size;
    static constexpr int axes = StaticConfigT::axes;

    DALI_HOST_DEV DALI_FORCEINLINE int flat_size() const {
      return threadblock_size;
    }

    DALI_HOST_DEV DALI_FORCEINLINE ivec<axes> block_dim() const {
      return cat(ivec<1>{threadblock_size}, ivec<axes - 1>{1});
    }

    DALI_HOST_DEV ivec<axes> log_block_dim() const {
      return block_dim() * StaticConfigT::lanes_dim();
    }

    DALI_DEVICE DALI_FORCEINLINE int flat_idx() const {
      return threadIdx.x;
    }

    DALI_DEVICE DALI_FORCEINLINE ivec<axes> thread_idx() const {
      return cat(ivec<1>{threadIdx.x}, ivec<axes - 1>{0});
    }
  };

  DALI_DEVICE DALI_FORCEINLINE BlockSetup operator()(int sample_idx) {
    (void)sample_idx;
    return {};
  }
};

template <typename StaticConfigT_>
struct AdaptiveBlock {
  struct BlockSetup {
    using StaticConfigT = StaticConfigT_;
    static constexpr int axes = StaticConfigT::axes;

    ivec<axes> extents_log2;
    ivec<axes> strides_log2;

    DALI_DEVICE DALI_FORCEINLINE int flat_size() const {
      assert(blockDim.y == 1);
      assert(blockDim.z == 1);
      return blockDim.x;
    }

    DALI_HOST_DEV DALI_FORCEINLINE ivec<axes> block_dim() const {
      return 1 << extents_log2;
    }

    DALI_HOST_DEV ivec<axes> log_block_dim() const {
      return block_dim() * StaticConfigT::lanes_dim();
    }

    DALI_DEVICE DALI_FORCEINLINE int flat_idx() const {
      return threadIdx.x;
    }

    DALI_DEVICE DALI_FORCEINLINE ivec<axes> thread_idx() const {
      assert(threadIdx.y == 0);
      assert(threadIdx.z == 0);
      return (int(threadIdx.x) >> strides_log2) & (block_dim() - 1);
    }
  };

  DALI_DEVICE DALI_FORCEINLINE const BlockSetup& operator()(int sample_idx) {
    return blocks[sample_idx];
  }

  const BlockSetup* blocks;
};

template <typename StaticConfigT>
std::enable_if_t<StaticConfigT::axes == 2, typename AdaptiveBlock<StaticConfigT>::BlockSetup>
create_adaptive_block(ivec2 extents_log2) {
  return {extents_log2, {0, extents_log2.x}};
}

template <typename StaticConfigT>
std::enable_if_t<StaticConfigT::axes == 3, typename AdaptiveBlock<StaticConfigT>::BlockSetup>
create_adaptive_block(ivec3 extents_log2) {
  return {extents_log2, {0, extents_log2.x, extents_log2.x + extents_log2.y}};
}

template <int axes>
struct ShapeDesc {
  int64_t frame_stride;       // (d)hwc
  i64vec<axes> in_strides;    // 1, wc(, hwc)
  int num_frames, width;      // f, w
  int num_channels;           // c
  ivec<axes> in_extents;      // wc, h(, d)
  ivec<axes> filter_extents;  // use workspace? rc, s(, p) : r, s(, p)
  ivec<axes> anchor_shift;    // anchor_r * c, anchor_s(, anchor_p)
  // threadblock * lanes + filter_extents - 1
  ivec<axes> in_workspace_extents;
  // the strides for in_workspace_extents
  ivec<axes> in_workspace_strides;
  // the sum of all but first in_workspace_extents
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
    if (idx >= sample_shape.in_extents.x) {
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
 * for the patch of logical block size (log_block_dim) and stores it in the provided ``acc``.
 * @{
 */

/**
 * @brief First loads the input patch necessary to compute the output of logical block size
 * (log_block_dim) into shared memory (including filter's halo/apron), then computes the
 * convolution.
 */
template <typename SampleDescT_, typename Inloader_, typename BlockSetupT_>
struct ShmInputConv {
  using SampleDescT = SampleDescT_;
  using Inloader = Inloader_;
  using BlockSetupT = BlockSetupT_;
  using StaticConfigT = typename BlockSetupT::StaticConfigT;
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
    const auto& thread_idx = block_setup.thread_idx();
    stride_filter(sample_desc.shape.filter_extents, [&](auto filter_coef, const auto& filter_offset,
                                                        int lane) {
      auto in_val =
          in_workspace[dot(thread_idx + filter_offset, sample_desc.shape.in_workspace_strides)];
      acc[lane] += in_val * filter_coef;
    });
  }

  template <typename MulAddCoef>
  DALI_DEVICE DALI_FORCEINLINE void stride_filter(
      const ivec2& filter_extents, MulAddCoef&& mul_add_coef) const {
    const auto& block_dim = block_setup.block_dim();
    const auto* filter = sample_desc.filter;
    for (int r = 0; r < filter_extents.y; r++) {
      for (int s = 0; s < filter_extents.x; s += sample_desc.shape.num_channels) {
        auto filter_coef = __ldg(filter++);
#pragma unroll
        for (int lane = 0, lanes_offset = 0; lane < StaticConfigT::lanes;
             lane++, lanes_offset += block_dim.y) {
          ivec2 filter_offset{s, r + lanes_offset};
          mul_add_coef(filter_coef, filter_offset, lane);
        }
      }
    }
  }

  template <typename MulAddCoef>
  DALI_DEVICE DALI_FORCEINLINE void stride_filter(
      const ivec3& filter_extents, MulAddCoef&& mul_add_coef) const {
    const auto& block_dim = block_setup.block_dim();
    const auto* filter = sample_desc.filter;
    for (int p = 0; p < filter_extents.z; p++) {
      for (int r = 0; r < filter_extents.y; r++) {
        for (int s = 0; s < filter_extents.x; s += sample_desc.shape.num_channels) {
          auto filter_coef = __ldg(filter++);
#pragma unroll
          for (int lane_z = 0, lane_z_offset = 0; lane_z < StaticConfigT::lanes_z;
               lane_z++, lane_z_offset += block_dim.z) {
#pragma unroll
            for (int lane_y = 0, lane_y_offset = 0; lane_y < StaticConfigT::lanes_y;
                 lane_y++, lane_y_offset += block_dim.y) {
              ivec3 filter_offset{s, r + lane_y_offset, p + lane_z_offset};
              mul_add_coef(filter_coef, filter_offset, lane_z * StaticConfigT::lanes_y + lane_y);
            }
          }
        }
      }
    }
  }

  DALI_DEVICE DALI_FORCEINLINE void precompute_indices(
      const In* __restrict__ in, const ivec2& anchored_start) const {
    for (int y = block_setup.flat_idx(); y < sample_desc.shape.in_workspace_extents.y;
         y += block_setup.flat_size()) {
      precomputed_idx[y] = in_loader.border_remap(anchored_start.y + y, sample_desc.shape, 1);
    }
  }

  DALI_DEVICE DALI_FORCEINLINE void precompute_indices(
      const In* __restrict__ in, const ivec3& anchored_start) const {
    int* ys = precomputed_idx;
    for (int y = block_setup.flat_idx(); y < sample_desc.shape.in_workspace_extents.y;
         y += block_setup.flat_size()) {
      ys[y] = in_loader.border_remap(anchored_start.y + y, sample_desc.shape, 1);
    }
    int* zs = precomputed_idx + sample_desc.shape.in_workspace_extents.y;
    for (int z = block_setup.flat_idx(); z < sample_desc.shape.in_workspace_extents.z;
         z += block_setup.flat_size()) {
      zs[z] = in_loader.border_remap(anchored_start.z + z, sample_desc.shape, 2);
    }
  }

  DALI_DEVICE DALI_FORCEINLINE void load_input_to_shm(
      const In* __restrict__ in, const ivec2& anchored_start) const {
    const auto& block_dim = block_setup.block_dim();
    const auto& thread_idx = block_setup.thread_idx();
    for (int x = thread_idx.x; x < sample_desc.shape.in_workspace_extents.x; x += block_dim.x) {
      int global_x = in_loader.border_remap_innermost(anchored_start.x + x, sample_desc.shape);
      for (int y = thread_idx.y; y < sample_desc.shape.in_workspace_extents.y; y += block_dim.y) {
        int global_y = precomputed_idx[y];
        in_workspace[dot(ivec2{x, y}, sample_desc.shape.in_workspace_strides)] =
            in_loader.load(in, ivec2{global_x, global_y}, sample_desc.shape);
      }
    }
  }

  DALI_DEVICE DALI_FORCEINLINE void load_input_to_shm(
      const In* __restrict__ in, const ivec3& anchored_start) const {
    const auto& block_dim = block_setup.block_dim();
    const auto& thread_idx = block_setup.thread_idx();
    for (int x = thread_idx.x; x < sample_desc.shape.in_workspace_extents.x; x += block_dim.x) {
      int global_x = in_loader.border_remap_innermost(anchored_start.x + x, sample_desc.shape);
      for (int y = thread_idx.y; y < sample_desc.shape.in_workspace_extents.y; y += block_dim.y) {
        int global_y = precomputed_idx[y];
        for (int z = thread_idx.z; z < sample_desc.shape.in_workspace_extents.z; z += block_dim.z) {
          int global_z = precomputed_idx[z];
          in_workspace[dot(ivec3{x, y, z}, sample_desc.shape.in_workspace_strides)] =
              in_loader.load(in, ivec3{global_x, global_y, global_z}, sample_desc.shape);
        }
      }
    }
  }

  const SampleDescT& sample_desc;
  const Inloader& in_loader;
  const BlockSetupT& block_setup;
  In* in_workspace;
  int* precomputed_idx;
};

template <typename SampleDescT, typename Inloader, typename BlockSetupT, typename In>
DALI_DEVICE DALI_FORCEINLINE ShmInputConv<SampleDescT, Inloader, BlockSetupT> create_shm_conv(
    const SampleDescT& sample_desc, const Inloader& in_loader, const BlockSetupT& block_setup,
    In* in_workspace, int* precomputed_idx) {
  return {sample_desc, in_loader, block_setup, in_workspace, precomputed_idx};
}

/**
 * @brief Computes the convolution of logical block size size (log_block_dim) accessing the input
 * directly in global memory. Used as a fallback when the filter size or number of channels in the
 * input make it impossible to use ``ShmInputConv``.
 */
template <typename SampleDescT_, typename Inloader_, typename BlockSetupT_>
struct DirectInputConv {
  using SampleDescT = SampleDescT_;
  using Inloader = Inloader_;
  using BlockSetupT = BlockSetupT_;
  using StaticConfigT = typename BlockSetupT::StaticConfigT;
  using Acc = typename SampleDescT::Acc;
  using In = typename SampleDescT::In;

  DALI_DEVICE DALI_FORCEINLINE void compute(Acc* __restrict__ acc, const In* __restrict__ in,
                                            const ivec2& start) const {
    const auto& block_dim = block_setup.block_dim();
    const auto& thread_idx = block_setup.thread_idx();
    auto start_shifted = start - sample_desc.shape.anchor_shift;
    const auto* filter = sample_desc.filter;
    for (int s = 0; s < sample_desc.shape.filter_extents.x; s++) {
      auto global_x = in_loader.border_remap_innermost(
          start_shifted.x + thread_idx.x + s * sample_desc.shape.num_channels, sample_desc.shape);
      for (int r = 0; r < sample_desc.shape.filter_extents.y; r++) {
        auto filter_coef = __ldg(filter + r * sample_desc.shape.filter_extents.x + s);
        // Even without shm, using `lanes` speeds up the kernel by reducing
        // the cost of nested loops arithmetic per single output value
#pragma unroll
        for (int lane = 0; lane < StaticConfigT::lanes; lane++) {
          auto global_y = in_loader.border_remap(
              start_shifted.y + thread_idx.y + lane * block_dim.y + r, sample_desc.shape, 1);
          auto in_val = in_loader.load(in, ivec2{global_x, global_y}, sample_desc.shape);
          acc[lane] += in_val * filter_coef;
        }
      }
    }
  }

  DALI_DEVICE DALI_FORCEINLINE void compute(Acc* __restrict__ acc, const In* __restrict__ in,
                                            const ivec3& start) const {
    const auto& block_dim = block_setup.block_dim();
    const auto& thread_idx = block_setup.thread_idx();
    auto start_shifted = start - sample_desc.shape.anchor_shift;
    const auto* filter = sample_desc.filter;
    for (int s = 0; s < sample_desc.shape.filter_extents.x; s++) {
      auto global_x = in_loader.border_remap_innermost(
          start_shifted.x + thread_idx.x + s * sample_desc.shape.num_channels, sample_desc.shape);
      for (int r = 0; r < sample_desc.shape.filter_extents.y; r++) {
        auto filter_coef = __ldg(filter + r * sample_desc.shape.filter_extents.x + s);
        // Even without shm, using `lanes` speeds up the kernel by reducing
        // the cost of nested loops arithmetic per single output value
#pragma unroll
        for (int lane = 0; lane < StaticConfigT::lanes; lane++) {
          auto global_y = in_loader.border_remap(
              start_shifted.y + thread_idx.y + lane * block_dim.y + r, sample_desc.shape, 1);
          auto in_val = in_loader.load(in, ivec2{global_x, global_y}, sample_desc.shape);
          acc[lane] += in_val * filter_coef;
        }
      }
    }
  }

  const SampleDescT& sample_desc;
  const Inloader& in_loader;
  const BlockSetupT& block_setup;
};


template <typename SampleDescT, typename Inloader, typename BlockSetupT>
DALI_DEVICE DALI_FORCEINLINE DirectInputConv<SampleDescT, Inloader, BlockSetupT> create_direct_conv(
    const SampleDescT& sample_desc, const Inloader& in_loader, const BlockSetupT& block_setup) {
  return {sample_desc, in_loader, block_setup};
}


/** @} */  // end of InputConv

template <typename BlockSetupT, typename Cb>
DALI_DEVICE DALI_FORCEINLINE void for_each_output_point_in_log_block(const ivec2& block_start,
                                                                     const ivec2& roi_size,
                                                                     const BlockSetupT& block_setup,
                                                                     const Cb&& cb) {
  using StaticConfigT = typename BlockSetupT::StaticConfigT;
  const auto& block_dim = block_setup.block_dim();
  auto coords = block_start + block_setup.thread_idx();
  if (coords.x < roi_size.x) {
#pragma unroll
    for (int lane = 0; lane < StaticConfigT::lanes; lane++) {
      if (coords.y < roi_size.y) {
        cb(coords, lane);
      }
      coords.y += block_dim.y;
    }
  }
}

template <typename BlockSetupT, typename Cb>
DALI_DEVICE DALI_FORCEINLINE void for_each_output_point_in_log_block(const ivec3& block_start,
                                                                     const ivec3& roi_size,
                                                                     const BlockSetupT& block_setup,
                                                                     const Cb&& cb) {
  using StaticConfigT = typename BlockSetupT::StaticConfigT;
  const auto& block_dim = block_setup.block_dim();
  auto coords = block_start + block_setup.thread_idx();
  if (coords.x < roi_size.x) {
#pragma unroll
    for (int lane_z = 0; lane_z < StaticConfigT::lanes_z; lane_z++) {
      if (coords.z < roi_size.z) {
#pragma unroll
        for (int lane_y = 0; lane_y < StaticConfigT::lanes_y; lane_y++) {
          if (coords.y < roi_size.y) {
            cb(coords, lane_z * StaticConfigT::lanes_y + lane_y);
          }
          coords.z += block_dim.z;
        }
      }
      coords.y += block_dim.y;
    }
  }
}

template <typename ROI, typename DoConv, typename In, typename Out>
DALI_DEVICE DALI_FORCEINLINE void stride_grid(const ivec2& block_start, const ivec2& grid_extents,
                                              const ROI& roi, const In* __restrict__ in,
                                              Out* __restrict__ out, DoConv&& do_conv) {
  const auto& roi_size = roi.size();
  for (int y_start = block_start.y; y_start < roi_size.y; y_start += grid_extents.y) {
    for (int x_start = block_start.x; x_start < roi_size.x; x_start += grid_extents.x) {
      do_conv(ivec2{x_start, y_start}, in, out);
    }
  }
}

template <typename ROI, typename DoConv, typename In, typename Out>
DALI_DEVICE DALI_FORCEINLINE void stride_grid(const ivec3& block_start, const ivec3& grid_extents,
                                              const ROI& roi, const In* __restrict__ in,
                                              Out* __restrict__ out, DoConv&& do_conv) {
  const auto& roi_size = roi.size();
  for (int z_start = block_start.z; z_start < roi_size.z; z_start += grid_extents.z) {
    for (int y_start = block_start.y; y_start < roi_size.y; y_start += grid_extents.y) {
      for (int x_start = block_start.x; x_start < roi_size.x; x_start += grid_extents.x) {
        do_conv(ivec3{x_start, y_start, z_start}, in, out);
      }
    }
  }
}

template <typename ROI, typename Conv, int axes>
DALI_DEVICE DALI_FORCEINLINE void stride_grid(const ivec<axes>& initial_block_start,
                                              const ivec<axes>& grid_extents, const ROI& roi,
                                              const Conv& conv) {
  using BlockSetupT = typename Conv::BlockSetupT;
  using StaticConfigT = typename BlockSetupT::StaticConfigT;
  using SampleDescT = typename Conv::SampleDescT;
  using Acc = typename SampleDescT::Acc;
  using Out = typename SampleDescT::Out;
  constexpr int lanes = StaticConfigT::lanes;
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
          for_each_output_point_in_log_block(
              block_start, roi.size(), conv.block_setup, [&](const auto& coords, int lane) {
                out[dot(coords, roi_strides)] = ConvertSat<Out>(acc[lane]);
              });
        });
  }
}

template <typename SampleDescT, typename InputROIFactory, typename BlockSetupFactory,
          typename InLoaderFactory, typename GridSetupT>
__global__ void filter(const SampleDescT* __restrict__ descs, InputROIFactory in_roi_factory,
                       BlockSetupFactory block_setup_factory, InLoaderFactory in_loader_factory,
                       GridSetupT grid_setup) {
  extern __shared__ char shm[];
  int sample_idx = grid_setup.sample_idx();
  auto sample_desc = descs[sample_idx];
  const auto& roi = in_roi_factory(sample_desc.shape, sample_idx);
  const auto& block_setup = block_setup_factory(sample_idx);
  auto block_start = grid_setup.block_idx() * block_setup.log_block_dim();
  if (any_coord(block_start >= roi.size())) {
    return;  // early exit to avoid all the setup only to do nothing in the stride loop
  }
  auto grid_size = grid_setup.grid_dim() * block_setup.log_block_dim();
  const auto& in_loader = in_loader_factory(sample_idx);
  if (sample_desc.shape.in_workspace_extents.x) {
    using In = typename SampleDescT::In;
    int* precomputed_idx = reinterpret_cast<int*>(shm);
    In* in_workspace = reinterpret_cast<In*>(shm + sample_desc.shape.in_workspace_offset);
    auto conv = create_shm_conv(sample_desc, in_loader, block_setup, in_workspace, precomputed_idx);
    stride_grid(block_start, grid_size, roi, conv);
  } else {
    auto conv = create_direct_conv(sample_desc, in_loader, block_setup);
    stride_grid(block_start, grid_size, roi, conv);
  }
}

template <int axes>
struct StaticConfig {};

template <>
struct StaticConfig<2> {
  static constexpr int threadblock_size = 128;
  static constexpr int axes = 2;
  static constexpr int lanes = 8;
  static constexpr int max_grid_extent = 32;

  static DALI_HOST_DEV ivec2 lanes_dim() {
    return {1, lanes};
  }

  static DALI_HOST_DEV ivec2 max_grid_extents() {
    return {max_grid_extent, max_grid_extent};
  }
};

template <>
struct StaticConfig<3> {
  static constexpr int threadblock_size = 128;
  static constexpr int axes = 3;
  static constexpr int lanes_y = 4;
  static constexpr int lanes_z = 4;
  static constexpr int lanes = lanes_y * lanes_z;
  static constexpr int max_grid_extent = 8;

  static DALI_HOST_DEV ivec3 lanes_dim() {
    return {1, lanes_y, lanes_z};
  }

  static DALI_HOST_DEV ivec3 max_grid_extents() {
    return {max_grid_extent, max_grid_extent, max_grid_extent};
  }
};

template <typename StaticConfigT>
struct GridSetup {
  static constexpr int axes = StaticConfigT::axes;

  DALI_HOST_DEV ivec<axes> grid_dim() const {
    return num_blocks_;
  }

  template <int axes_ = axes>
  DALI_DEVICE std::enable_if_t<axes_ == 2, ivec2> block_idx() const {
    return {blockIdx.x, blockIdx.y};
  }

  template <int axes_ = axes>
  DALI_DEVICE std::enable_if_t<axes_ == 3, ivec3> block_idx() const {
    return {blockIdx.x, blockIdx.y & (StaticConfigT::max_grid_extent - 1),
            blockIdx.y / StaticConfigT::max_grid_extent};
  }

  template <int axes_ = axes>
  std::enable_if_t<axes_ == 2, dim3> kernel_setup() const {
    auto grid_dim = this->grid_dim();
    return {static_cast<unsigned int>(grid_dim.x), static_cast<unsigned int>(grid_dim.y),
            static_cast<unsigned int>(num_samples())};
  }

  template <int axes_ = axes>
  std::enable_if_t<axes_ == 3, dim3> kernel_setup() const {
    auto grid_dim = this->grid_dim();
    return {static_cast<unsigned int>(grid_dim.x),
            static_cast<unsigned int>(grid_dim.y * StaticConfigT::max_grid_extents + grid_dim.z),
            static_cast<unsigned int>(num_samples())};
  }

  DALI_DEVICE int sample_idx() const {
    return blockIdx.z;
  }

  DALI_HOST_DEV int num_samples() const {
    return num_samples_;
  }

  ivec<axes> num_blocks_;
  int num_samples_;
};

}  // namespace filter

template <typename Out, typename In, typename W, bool has_channel_dim, bool has_sequence_dim,
          int axes_>
struct FilterGpu {
  /* It computes a correlation of the input and the filter.
  Flip filter in both dimensions for a convolution. */

  static constexpr int axes = axes_;
  static constexpr int ndim =
      static_cast<int>(has_sequence_dim) + axes + static_cast<int>(has_channel_dim);
  static constexpr int sequence_dim = has_sequence_dim ? 0 : -1;
  static constexpr int channels_dim = has_channel_dim ? ndim - 1 : -1;
  using Intermediate = decltype(std::declval<W>() * std::declval<In>());
  using StaticConfigT = filter::StaticConfig<axes>;
  using BlockSetupFactoryT = filter::AdaptiveBlock<StaticConfigT>;
  using BlockSetupT = typename BlockSetupFactoryT::BlockSetup;
  using StaticBlockFactoryT = filter::StaticBlock<StaticConfigT>;
  using StaticBlockT = typename StaticBlockFactoryT::BlockSetup;
  using GridSetupT = filter::GridSetup<StaticConfigT>;
  using SampleDescT = filter::SampleDesc<Out, In, W, Intermediate, axes>;
  using CustomROIFactoryT = typename filter::CustomInputROI<axes>;
  using CustomROIT = typename CustomROIFactoryT::ROI;

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

    int max_total_workspace;
    bool any_has_degenerated_extents;
    SampleDescT* samples_desc_dev;
    SetupSampleDescs(max_total_workspace, any_has_degenerated_extents, out, in, filters, anchors);
    std::tie(samples_desc_dev) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, samples_desc_);
    auto grid_setup = PrepareGridSetup(out, in);
    RunKernel(ctx, out.shape, in.shape, input_rois, border_type, fill_values,
              any_has_degenerated_extents, [&](auto&& roi) {
                return [&](auto&& block_setup) {
                  return [&](auto&& loader) {
                    filter::filter<<<grid_setup.kernel_setup(), StaticConfigT::threadblock_size,
                                     max_total_workspace, ctx.gpu.stream>>>(
                        samples_desc_dev, roi, block_setup, loader, grid_setup);
                    CUDA_CALL(cudaGetLastError());
                  };
                };
              });
  }

 protected:
  template <typename T>
  vec<axes, T> ShapeAsVec(const TensorShape<ndim>& shape) {
    return shape2vec<axes, T>(skip_dim<sequence_dim>(skip_dim<channels_dim>(shape)));
  }

  template <typename Inshapes, typename FilterShapes>
  void ValidateNumericLimits(const Inshapes& in_shapes, const FilterShapes& filter_shapes,
                             const span<const ivec<axes>> anchors) {
    const auto max_grid_logical_extents =
        StaticConfigT::max_grid_extents() * StaticConfigT::lanes_dim();
    // so that we can safely use grid extents as a stride in a for loop
    const auto max_sample_extents = std::numeric_limits<int>::max() - max_grid_logical_extents + 1;
    for (int sample_idx = 0; sample_idx < in_shapes.num_samples(); sample_idx++) {
      const auto& in_shape = in_shapes[sample_idx];
      const auto& filter_shape = filter_shapes[sample_idx];
      const auto& anchor = anchors[sample_idx];
      int64_t num_frames = has_sequence_dim ? in_shape[sequence_dim] : 1;
      int64_t num_channels = has_channel_dim ? in_shape[channels_dim] : 1;
      const auto channels = cat(i64vec<1>{num_channels}, i64vec<axes - 1>{1});
      auto in_extents = ShapeAsVec<int64_t>(in_shape) * channels;
      auto filter_extents = shape2vec<axes, int64_t>(filter_shape);
      DALI_ENFORCE(
          num_frames <= std::numeric_limits<int>::max(),
          make_string("Number of frames for sample of idx ", sample_idx, " exceeds the limit of ",
                      std::numeric_limits<int>::max(), ". Got: ", num_frames, "."));
      DALI_ENFORCE(
          volume(filter_extents) <= std::numeric_limits<int>::max(),
          make_string("Volume of filter for sample of idx ", sample_idx, " exceeds the limit of ",
                      std::numeric_limits<int>::max(), ". Got: ", volume(filter_extents), "."));
      DALI_ENFORCE(all_coords(in_extents <= static_cast<i64vec<axes>>(max_sample_extents)),
                   make_string("The size of the sample of idx ", sample_idx, " is ", in_extents,
                               ", which exceeds the limit of ", max_sample_extents, "."));
    }
  }

  void SetupSampleDescs(int& max_total_workspace, bool& any_has_degenerated_extents,
                        const TensorListView<StorageGPU, Out, ndim>& out,
                        const TensorListView<StorageGPU, const In, ndim>& in,
                        const TensorListView<StorageGPU, const W, axes>& filters,
                        const span<const ivec<axes>> anchors) {
    const auto& in_shapes = in.shape;
    const auto& filter_shapes = filters.shape;
    int num_samples = in_shapes.num_samples();
    ValidateNumericLimits(in_shapes, filter_shapes, anchors);
    samples_desc_.clear();
    samples_desc_.reserve(num_samples);
    block_setups_.clear();
    block_setups_.reserve(num_samples);
    max_total_workspace = 0;
    any_has_degenerated_extents = false;
    const int shared_mem_limit = GetSharedMemPerBlock();
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      int required_workspace;
      bool has_degenerated_extents;
      BlockSetupT block_setup;
      auto shape_desc = SetupSampleShapeDesc(
          required_workspace, has_degenerated_extents, block_setup, sample_idx,
          in_shapes[sample_idx], filter_shapes[sample_idx], anchors[sample_idx], shared_mem_limit);
      any_has_degenerated_extents |= has_degenerated_extents;
      max_total_workspace = std::max(max_total_workspace, required_workspace);
      block_setups_.push_back(block_setup);
      samples_desc_.push_back({out.tensor_data(sample_idx), in.tensor_data(sample_idx),
                               filters.tensor_data(sample_idx), shape_desc});
    }
  }

  template <typename InShape, typename FilterShape>
  filter::ShapeDesc<axes> SetupSampleShapeDesc(int& required_workspace,
                                               bool& has_degenerated_extents,
                                               BlockSetupT& block_setup, int sample_idx,
                                               const InShape& in_shape,
                                               const FilterShape& filter_shape, ivec<axes> anchor,
                                               int shared_mem_limit) {
    int num_frames = has_sequence_dim ? in_shape[sequence_dim] : 1;
    int num_channels = has_channel_dim ? in_shape[channels_dim] : 1;
    const auto channels = cat(ivec<1>{num_channels}, ivec<axes - 1>{1});
    auto in_extents = ShapeAsVec<int>(in_shape);
    int width = in_extents.x;
    auto filter_extents = shape2vec(filter_shape);
    block_setup = PrepareBlockSetup(in_extents * channels);
    auto log_block_dim = block_setup.log_block_dim();
    auto in_workspace_extents = (filter_extents - 1) * channels + log_block_dim;
    auto in_workspace_num_elements = volume(static_cast<i64vec<axes>>(in_workspace_extents));
    auto idx_workspace_size =
        std::accumulate(in_workspace_extents.begin() + 1, in_workspace_extents.end(), 0);
    int in_workspace_offset = align_up(idx_workspace_size * sizeof(int), sizeof(In));
    required_workspace = in_workspace_offset + in_workspace_num_elements * sizeof(In);
    if (num_channels > log_block_dim.x || required_workspace > shared_mem_limit) {
      in_workspace_extents = required_workspace = 0;
    }
    has_degenerated_extents = any_coord(in_extents <= 1);
    int64_t frame_stride;
    i64vec<axes> in_strides;
    filter::strides(in_strides, frame_stride, in_extents * channels);
    ivec<axes> workspace_strides;
    filter::strides(workspace_strides, in_workspace_extents);
    return {frame_stride,
            in_strides,
            num_frames,
            width,
            num_channels,
            in_extents * channels,
            required_workspace == 0 ? filter_extents : filter_extents * channels,
            anchor * channels,
            in_workspace_extents,
            workspace_strides,
            in_workspace_offset};
  }

  BlockSetupT PrepareBlockSetup(ivec<axes> in_extents) {
    int max_block_width_log2 = dali::ilog2(StaticConfigT::threadblock_size);
    int sample_wc_log2 = in_extents.x == 0 ? 0 : dali::ilog2(in_extents.x - 1) + 1;
    int block_width_log2 = std::min(max_block_width_log2, sample_wc_log2);
    return filter::create_adaptive_block<StaticConfigT>(
        {block_width_log2, max_block_width_log2 - block_width_log2});
  }

  GridSetupT PrepareGridSetup(const TensorListView<StorageGPU, Out, ndim>& out,
                              const TensorListView<StorageGPU, const In, ndim>& in) {
    ivec<axes> num_blocks = 0;
    const auto& in_shapes = in.shape;
    const auto& out_shapes = out.shape;
    for (int sample_idx = 0; sample_idx < in_shapes.num_samples(); sample_idx++) {
      const auto& out_shape = out_shapes[sample_idx];
      int64_t num_channels = has_channel_dim ? in_shapes[sample_idx][channels_dim] : 1;
      const auto channels = cat(ivec<1>{num_channels}, ivec<axes - 1>{1});
      auto out_extents = ShapeAsVec<int>(out_shape) * channels;
      auto sample_num_blocks = div_ceil(out_extents, block_setups_[sample_idx].log_block_dim());
      num_blocks = max(num_blocks, sample_num_blocks);
    }
    num_blocks = min(num_blocks, StaticConfigT::max_grid_extents());
    return {num_blocks, in_shapes.num_samples()};
  }

  template <typename Shape>
  CustomROIT SetupValidateROI(const Shape& out_shape, const Shape& in_shape,
                              const filter::ShapeDesc<axes>& shape_desc,
                              const filter::InputROI<axes>& roi) {
    int64_t num_channels = has_channel_dim ? in_shape[channels_dim] : 1;
    const auto channels = cat(ivec<1>{num_channels}, ivec<axes - 1>{1});
    const auto out_extents = ShapeAsVec<int>(out_shape);
    const auto in_extents = ShapeAsVec<int>(in_shape);
    auto roi_extents = roi.end - roi.start;
    DALI_ENFORCE(
        roi_extents == out_extents,
        make_string("The output size must match the input roi size. Got output of size: ",
                    filter::rev(out_extents), " and roi of size ", filter::rev(roi_extents), "."));
    DALI_ENFORCE(all_coords(0 <= roi.start) && all_coords(roi.end <= in_extents),
                 make_string("ROI must lie within the input sample. Got roi that starts at: ",
                             roi.start, " and ends at ", filter::rev(roi.end),
                             " for a sample of shape ", filter::rev(in_extents), "."));
    auto roi_start = roi.start * channels;
    roi_extents *= channels;
    int64_t frame_roi_stride;
    i64vec<axes> roi_strides;
    filter::strides(roi_strides, frame_roi_stride, roi_extents);
    return {roi_start, roi_extents, roi_strides, frame_roi_stride};
  }

  template <typename InShapes, typename OutShapes, typename KernelLauncher>
  void RunKernel(KernelContext& ctx, const OutShapes& out_shapes, const InShapes& in_shapes,
                 const span<const filter::InputROI<axes>> input_rois,
                 boundary::BoundaryType border_type,
                 const TensorListView<StorageGPU, const In, 0>& fill_values,
                 bool has_degenerated_extents, KernelLauncher&& launch_kernel) {
    if (input_rois.size() == 0) {
      filter::InputRoiFull<axes> roi_handler{};
      RunKernelWithBlockSetup(ctx, border_type, fill_values, has_degenerated_extents,
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
      RunKernelWithBlockSetup(ctx, border_type, fill_values, has_degenerated_extents,
                              launch_kernel(std::move(roi_handler)));
    }
  }

  template <typename KernelLauncher>
  void RunKernelWithBlockSetup(KernelContext& ctx, boundary::BoundaryType border_type,
                               const TensorListView<StorageGPU, const In, 0>& fill_values,
                               bool has_degenerated_extents, KernelLauncher&& launch_kernel) {
    if (std::all_of(block_setups_.begin(), block_setups_.end(), [](const auto& block_setup) {
          return block_setup.block_dim() == StaticBlockT{}.block_dim();
        })) {
      RunKernelWithBorderMode(ctx, border_type, fill_values, has_degenerated_extents,
                              launch_kernel(StaticBlockFactoryT{}));
    } else {
      BlockSetupT* block_setups_dev;
      std::tie(block_setups_dev) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, block_setups_);
      BlockSetupFactoryT block_setup{block_setups_dev};
      RunKernelWithBorderMode(ctx, border_type, fill_values, has_degenerated_extents,
                              launch_kernel(std::move(block_setup)));
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

  std::vector<SampleDescT> samples_desc_;
  std::vector<const In*> fill_values_;
  std::vector<CustomROIT> custom_rois_;
  std::vector<BlockSetupT> block_setups_;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_FILTER_GPU_H_
