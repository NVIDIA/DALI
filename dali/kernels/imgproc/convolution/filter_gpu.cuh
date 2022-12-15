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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_FILTER_GPU_CUH_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_FILTER_GPU_CUH_

#include <array>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "dali/core/boundary.h"
#include "dali/core/convert.h"
#include "dali/core/cuda_rt_utils.h"
#include "dali/core/geom/geom_utils.h"
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

using boundary::BoundaryType;

template <int axes_>
struct InShapeDesc {
  static constexpr int axes = axes_;

  int64_t frame_stride;       // (d)hwc
  i64vec<axes> in_strides;    // 1, wc(, hwc)
  int num_frames, width;      // f, w
  int num_channels;           // c
  int in_filter_width;        // sc, i.e. innermost filter extent * num_channels
  ivec<axes> in_extents;      // wc, h(, d)
  ivec<axes> filter_extents;  // s, r(, p)
  ivec<axes> filter_strides;  // 1, r(, rs)
  ivec<axes> anchor_shift;    // anchor_s * c, anchor_r(, anchor_p)
};

template <int axes>
struct OutShapeDesc {
  int64_t frame_stride;  // (d)hwc
  i64vec<axes> strides;  // 1, wc(, hwc)
  ivec<axes> extents;    // wc, h(, d)
};

template <int axes>
struct WorkspaceDesc {
  // threadblock * lanes + {in_filter_width, r(, p)} - 1
  ivec<axes> in_extents;
  // the strides for the in_extents
  ivec<axes> in_strides;
  // The offset in shared memory that should be left
  // for precomuted indices. At that offset, the
  // input data will be loaded.
  // The offset is a sum of all but first in_extents
  // rounded up to the proper aligment for input data type
  int in_offset;
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
  InShapeDesc<axes> in_shape;
  OutShapeDesc<axes> out_shape;
  WorkspaceDesc<axes> workspace_desc;
  // the size of a logical block: threablock and lanes
  ivec<axes> logical_block_extents;
  // axis to iterate over with lanes
  // if axes == 2 then must be 1 (i.e. height),
  // for volumetric input it may be depth or height
  int lanes_axis;
};

/** @defgroup InputLoader InputLoader is meant to specialize loading of the sample from global
 * memory. This is how different border modes (apart from BORDER_VALID) are handled.
 * @{
 */
template <typename In, int axes, BoundaryType border>
struct InLoaderBorderRemap {
  template <BoundaryType border_>
  using BorderTag = std::integral_constant<BoundaryType, border_>;

  DALI_HOST_DEV DALI_FORCEINLINE int border_remap(int idx, int len,
                                                  BorderTag<BoundaryType::REFLECT_101>) const {
    assert(len > 0);
    return boundary::idx_reflect_101(idx, len);
  }

  DALI_HOST_DEV DALI_FORCEINLINE int border_remap(int idx, int len,
                                                  BorderTag<BoundaryType::REFLECT_1001>) const {
    assert(len > 0);
    return boundary::idx_reflect_1001(idx, len);
  }

  DALI_HOST_DEV DALI_FORCEINLINE int border_remap(int idx, int len,
                                                  BorderTag<BoundaryType::CLAMP>) const {
    assert(len > 0);
    return boundary::idx_clamp(idx, 0, len);
  }

  DALI_HOST_DEV DALI_FORCEINLINE int border_remap(int idx, int len,
                                                  BorderTag<BoundaryType::WRAP>) const {
    assert(len > 0);
    return boundary::idx_wrap(idx, len);
  }

  DALI_HOST_DEV DALI_FORCEINLINE int border_remap(int idx, const InShapeDesc<axes>& sample_shape,
                                                  int axis) const {
    return border_remap(idx, sample_shape.in_extents[axis], BorderTag<border>{});
  }

  DALI_HOST_DEV DALI_FORCEINLINE int border_remap_innermost(
      int idx, const InShapeDesc<axes>& sample_shape) const {
    int num_channels = sample_shape.num_channels;
    if (idx < 0) {
      // First, shift by 1 towards 0 (idx + 1), so that indices belonging to the same pixel
      // translate to the same number pixel index when dividing by the channels stride
      // (-6, -5, -4, -3, -2, -1) + 1 -> (-5, -4, -3, -2, -1, 0)
      // (-5, -4, -3, -2, -1, 0) / 3 -> (-1, -1, -1, 0, 0, 0)
      // Then shift back away from 0 (-1, -1, -1, 0, 0, 0) - 1 -> (-2, -2, -2, -1, -1, -1)
      // Finally, with (num_channels - 1) we get the positive channels indecies
      // (-2, -1, 0, -2, -1, 0) + 3 - 1 -> (0, 1, 2, 0, 1, 2)
      int reflect_dim_idx = (idx + 1) / num_channels - 1;
      int inner_dim_idx = (idx + 1) % num_channels + num_channels - 1;
      return border_remap(reflect_dim_idx, sample_shape.width, BorderTag<border>{}) * num_channels +
             inner_dim_idx;
    }
    if (idx >= sample_shape.in_extents.x) {
      return border_remap(idx / num_channels, sample_shape.width, BorderTag<border>{}) *
                 num_channels +
             idx % num_channels;
    }
    return idx;
  }

  DALI_HOST_DEV DALI_FORCEINLINE In load(const In* __restrict__ in, const ivec<axes>& coords,
                                         const InShapeDesc<axes>& sample_shape) const {
    return in[dot(coords, sample_shape.in_strides)];
  }
};

template <typename In, int axes>
struct InLoaderPad {
  DALI_HOST_DEV DALI_FORCEINLINE int border_remap(int idx, const InShapeDesc<axes>& sample_shape,
                                                  int axis) const {
    (void)sample_shape;
    (void)axis;
    return idx;
  }

  DALI_HOST_DEV DALI_FORCEINLINE int border_remap_innermost(
      int idx, const InShapeDesc<axes>& sample_shape) const {
    (void)sample_shape;
    return idx;
  }

  DALI_HOST_DEV DALI_FORCEINLINE In load(const In* __restrict__ in, const ivec<axes>& coords,
                                         const InShapeDesc<axes>& sample_shape) const {
    if (any_coord(coords < 0) || any_coord(coords >= sample_shape.in_extents)) {
      return fill_value;
    }
    return in[dot(coords, sample_shape.in_strides)];
  }

  In fill_value;
};

template <typename InLoader>
struct InLoaderProvider {
  DALI_DEVICE DALI_FORCEINLINE InLoader operator()(int sample_idx) {
    return {};
  }
};

template <typename In, int axes>
struct InLoaderProvider<InLoaderPad<In, axes>> {
  DALI_DEVICE DALI_FORCEINLINE InLoaderPad<In, axes> operator()(int sample_idx) {
    if (fill_values == nullptr) {
      return {0};
    }
    return {fill_values[sample_idx][0]};
  }
  const In** fill_values;
};

/** @} */  // end of InputLoader


/** @defgroup OutputShapeProvider OutputShapeProvider is introduced to reduce registers pressure
 * when it is known that input and output shapes are equal.
 * @{
 */

template <typename SampleDescT>
struct OutShapeProviderSame {
  DALI_DEVICE DALI_FORCEINLINE OutShapeDesc<SampleDescT::axes> operator()(SampleDescT sample_desc) {
    auto in_shape = sample_desc.in_shape;
    return {in_shape.frame_stride, in_shape.in_strides, in_shape.in_extents};
  }
};

template <typename SampleDescT>
struct OutShapeProviderROI {
  DALI_DEVICE DALI_FORCEINLINE OutShapeDesc<SampleDescT::axes> operator()(SampleDescT sample_desc) {
    return sample_desc.out_shape;
  }
};

/** @} */  // end of OutputShapeProvider


/** @defgroup InputConv The specializations provide ``compute`` method that computes convolution
 * for the patch of logical block size (logical_block_extents) and stores it in the provided
 * ``acc``.
 * @{
 */

/**
 * @brief First loads the input patch necessary to compute the output of logical block size
 * (logical_block_extents) into shared memory (including filter's halo/apron), then computes the
 * convolution.
 */
template <int lanes_axis_, typename SampleDescT_, typename Inloader_, typename BlockSetupT_>
struct ShmInputConv {
  using SampleDescT = SampleDescT_;
  using Inloader = Inloader_;
  using BlockSetupT = BlockSetupT_;
  using StaticConfigT = typename BlockSetupT::StaticConfigT;
  using In = typename SampleDescT::In;
  using Acc = typename SampleDescT::Acc;
  static constexpr int lanes_axis = lanes_axis_;

  DALI_DEVICE DALI_FORCEINLINE void compute(Acc* __restrict__ acc, const In* __restrict__ in,
                                            const ivec<SampleDescT::axes>& start) const {
    auto anchored_start = start - sample_desc.in_shape.anchor_shift;
    __syncthreads();
    precompute_indices(in, anchored_start);
    __syncthreads();
    load_input_to_shm(in, anchored_start);
    __syncthreads();
    const auto& thread_idx = block_setup.thread_idx();
    stride_filter(
        sample_desc.in_shape.filter_extents, sample_desc.in_shape.in_filter_width,
        [&](auto filter_coef, const auto& filter_offset, int lane) {
          auto in_val =
              in_workspace[dot(thread_idx + filter_offset, sample_desc.workspace_desc.in_strides)];
          acc[lane] += in_val * filter_coef;
        });
  }

  template <typename MulAddCoef>
  DALI_DEVICE DALI_FORCEINLINE void stride_filter(const ivec2& filter_extents, int in_filter_width,
                                                  MulAddCoef&& mul_add_coef) const {
    const auto& block_dim = block_setup.block_dim();
    const auto* filter = sample_desc.filter;
    for (int r = 0; r < filter_extents.y; r++) {
      for (int s = 0; s < in_filter_width; s += sample_desc.in_shape.num_channels) {
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
  DALI_DEVICE DALI_FORCEINLINE void stride_filter(const ivec3& filter_extents, int in_filter_width,
                                                  MulAddCoef&& mul_add_coef) const {
    const auto& block_dim = block_setup.block_dim();
    const auto* filter = sample_desc.filter;
    for (int p = 0; p < filter_extents.z; p++) {
      for (int r = 0; r < filter_extents.y; r++) {
        for (int s = 0; s < in_filter_width; s += sample_desc.in_shape.num_channels) {
          auto filter_coef = __ldg(filter++);
          ivec3 filter_position{s, r, p};
#pragma unroll
          for (int lane = 0; lane < StaticConfigT::lanes; lane++) {
            mul_add_coef(filter_coef, filter_position, lane);
            filter_position[lanes_axis] += block_dim[lanes_axis];
          }
        }
      }
    }
  }

  DALI_DEVICE DALI_FORCEINLINE void precompute_indices(const In* __restrict__ in,
                                                       const ivec2& anchored_start) const {
    for (int y = block_setup.flat_idx(); y < sample_desc.workspace_desc.in_extents.y;
         y += block_setup.flat_size()) {
      precomputed_idx[y] = in_loader.border_remap(anchored_start.y + y, sample_desc.in_shape, 1);
    }
  }

  DALI_DEVICE DALI_FORCEINLINE void precompute_indices(const In* __restrict__ in,
                                                       const ivec3& anchored_start) const {
    int* ys = precomputed_idx;
    for (int y = block_setup.flat_idx(); y < sample_desc.workspace_desc.in_extents.y;
         y += block_setup.flat_size()) {
      ys[y] = in_loader.border_remap(anchored_start.y + y, sample_desc.in_shape, 1);
    }
    int* zs = precomputed_idx + sample_desc.workspace_desc.in_extents.y;
    for (int z = block_setup.flat_idx(); z < sample_desc.workspace_desc.in_extents.z;
         z += block_setup.flat_size()) {
      zs[z] = in_loader.border_remap(anchored_start.z + z, sample_desc.in_shape, 2);
    }
  }

  DALI_DEVICE DALI_FORCEINLINE void load_input_to_shm(const In* __restrict__ in,
                                                      const ivec2& anchored_start) const {
    const auto& block_dim = block_setup.block_dim();
    const auto& thread_idx = block_setup.thread_idx();
    for (int x = thread_idx.x; x < sample_desc.workspace_desc.in_extents.x; x += block_dim.x) {
      int global_x = in_loader.border_remap_innermost(anchored_start.x + x, sample_desc.in_shape);
      for (int y = thread_idx.y; y < sample_desc.workspace_desc.in_extents.y; y += block_dim.y) {
        int global_y = precomputed_idx[y];
        in_workspace[dot(ivec2{x, y}, sample_desc.workspace_desc.in_strides)] =
            in_loader.load(in, ivec2{global_x, global_y}, sample_desc.in_shape);
      }
    }
  }

  DALI_DEVICE DALI_FORCEINLINE void load_input_to_shm(const In* __restrict__ in,
                                                      const ivec3& anchored_start) const {
    const int* const __restrict__ ys = precomputed_idx;
    const int* const __restrict__ zs = precomputed_idx + sample_desc.workspace_desc.in_extents.y;
    const auto& block_dim = block_setup.block_dim();
    const auto& thread_idx = block_setup.thread_idx();
    for (int x = thread_idx.x; x < sample_desc.workspace_desc.in_extents.x; x += block_dim.x) {
      int global_x = in_loader.border_remap_innermost(anchored_start.x + x, sample_desc.in_shape);
      for (int y = thread_idx.y; y < sample_desc.workspace_desc.in_extents.y; y += block_dim.y) {
        int global_y = ys[y];
        for (int z = thread_idx.z; z < sample_desc.workspace_desc.in_extents.z; z += block_dim.z) {
          int global_z = zs[z];
          in_workspace[dot(ivec3{x, y, z}, sample_desc.workspace_desc.in_strides)] =
              in_loader.load(in, ivec3{global_x, global_y, global_z}, sample_desc.in_shape);
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

/**
 * @brief Computes the convolution of logical block size size (logical_block_extents) accessing
 * the input directly in global memory. Used as a fallback when the filter size or number of
 * channels in the input make it impossible to use ``ShmInputConv``.
 */
template <int lanes_axis_, typename SampleDescT_, typename Inloader_, typename BlockSetupT_>
struct DirectInputConv {
  using SampleDescT = SampleDescT_;
  using Inloader = Inloader_;
  using BlockSetupT = BlockSetupT_;
  using StaticConfigT = typename BlockSetupT::StaticConfigT;
  using Acc = typename SampleDescT::Acc;
  using In = typename SampleDescT::In;
  static constexpr int lanes_axis = lanes_axis_;

  DALI_DEVICE DALI_FORCEINLINE void compute(Acc* __restrict__ acc, const In* __restrict__ in,
                                            const ivec2& start) const {
    const auto& block_dim = block_setup.block_dim();
    const auto& thread_idx = block_setup.thread_idx();
    const auto coords = start - sample_desc.in_shape.anchor_shift + thread_idx;
    const auto* filter = sample_desc.filter;
    for (int s = 0; s < sample_desc.in_shape.filter_extents.x; s++) {
      auto global_x = in_loader.border_remap_innermost(
          coords.x + s * sample_desc.in_shape.num_channels, sample_desc.in_shape);
      for (int r = 0; r < sample_desc.in_shape.filter_extents.y; r++) {
        auto filter_coef = __ldg(filter + dot(ivec2{s, r}, sample_desc.in_shape.filter_strides));
        // Even without shm, using `lanes` speeds up the kernel by reducing
        // the cost of nested loops arithmetic per single output value
#pragma unroll
        for (int lane = 0; lane < StaticConfigT::lanes; lane++) {
          auto global_y =
              in_loader.border_remap(coords.y + r + lane * block_dim.y, sample_desc.in_shape, 1);
          auto in_val = in_loader.load(in, ivec2{global_x, global_y}, sample_desc.in_shape);
          acc[lane] += in_val * filter_coef;
        }
      }
    }
  }

  DALI_DEVICE DALI_FORCEINLINE void compute(Acc* __restrict__ acc, const In* __restrict__ in,
                                            const ivec3& start) const {
    constexpr int non_lanes_axis = lanes_axis == 1 ? 2 : 1;
    const auto& block_dim = block_setup.block_dim();
    const auto& thread_idx = block_setup.thread_idx();
    const auto& filter_extents = sample_desc.in_shape.filter_extents;
    const auto in_coords = start - sample_desc.in_shape.anchor_shift + thread_idx;
    const auto* filter = sample_desc.filter;
    for (int s = 0; s < filter_extents.x; s++) {
      int global_x = in_loader.border_remap_innermost(
          in_coords.x + s * sample_desc.in_shape.num_channels, sample_desc.in_shape);
      for (int i = 0; i < filter_extents[non_lanes_axis]; i++) {
        int global_non_lanes_axis = in_loader.border_remap(in_coords[non_lanes_axis] + i,
                                                           sample_desc.in_shape, non_lanes_axis);
        for (int j = 0; j < filter_extents[lanes_axis]; j++) {
          ivec3 filter_pos{s, lanes_axis == 1 ? j : i, lanes_axis == 1 ? i : j};
          auto filter_coef = __ldg(filter + dot(filter_pos, sample_desc.in_shape.filter_strides));
          int lanes_axis_pos = in_coords[lanes_axis] + j;
          // Even without shm, using `lanes` speeds up the kernel by reducing
          // the cost of nested loops arithmetic per single output value
#pragma unroll
          for (int lane = 0; lane < StaticConfigT::lanes; lane++) {
            int global_lanes_axis = in_loader.border_remap(
                lanes_axis_pos + lane * block_dim[lanes_axis], sample_desc.in_shape, lanes_axis);
            ivec3 coords{global_x, lanes_axis == 1 ? global_lanes_axis : global_non_lanes_axis,
                         lanes_axis == 1 ? global_non_lanes_axis : global_lanes_axis};
            auto in_val = in_loader.load(in, coords, sample_desc.in_shape);
            acc[lane] += in_val * filter_coef;
          }
        }
      }
    }
  }

  const SampleDescT& sample_desc;
  const Inloader& in_loader;
  const BlockSetupT& block_setup;
};


struct ShmConvFactory {
  template <int lanes_axis, typename SampleDescT, typename Inloader, typename BlockSetupT>
  DALI_DEVICE DALI_FORCEINLINE ShmInputConv<lanes_axis, SampleDescT, Inloader, BlockSetupT> create(
      const SampleDescT& sample_desc, const Inloader& in_loader, const BlockSetupT& block_setup,
      char* shm) const {
    using In = typename SampleDescT::In;
    int* idx_workspace = reinterpret_cast<int*>(shm);
    In* in_workspace = reinterpret_cast<In*>(shm + sample_desc.workspace_desc.in_offset);
    return {sample_desc, in_loader, block_setup, in_workspace, idx_workspace};
  }
};

struct DirectConvFactory {
  template <int lanes_axis, typename SampleDescT, typename Inloader, typename BlockSetupT>
  DALI_DEVICE DALI_FORCEINLINE DirectInputConv<lanes_axis, SampleDescT, Inloader, BlockSetupT>
  create(const SampleDescT& sample_desc, const Inloader& in_loader, const BlockSetupT& block_setup,
         char* shm) const {
    (void)shm;
    return {sample_desc, in_loader, block_setup};
  }
};

template <typename ConvFactoryT, typename SampleDescT, typename InLoaderT, typename BlockSetupT,
          typename Cb>
DALI_DEVICE DALI_FORCEINLINE std::enable_if_t<SampleDescT::axes == 2, void> with_conv(
    const ConvFactoryT& conv_factory, const SampleDescT& sample_desc, const InLoaderT& in_loader,
    const BlockSetupT& block_setup, char* shm, Cb&& cb) {
  cb(conv_factory.create<1>(sample_desc, in_loader, block_setup, shm));
}

template <typename ConvFactoryT, typename SampleDescT, typename InLoaderT, typename BlockSetupT,
          typename Cb>
DALI_DEVICE DALI_FORCEINLINE std::enable_if_t<SampleDescT::axes == 3, void> with_conv(
    const ConvFactoryT& conv_factory, const SampleDescT& sample_desc, const InLoaderT& in_loader,
    const BlockSetupT& block_setup, char* shm, Cb&& cb) {
  if (sample_desc.lanes_axis == 2) {
    cb(conv_factory.create<2>(sample_desc, in_loader, block_setup, shm));
  } else {
    assert(sample_desc.lanes_axis == 1);
    cb(conv_factory.create<1>(sample_desc, in_loader, block_setup, shm));
  }
}

/** @} */  // end of InputConv

template <typename Conv, typename Cb>
DALI_DEVICE DALI_FORCEINLINE void for_each_output_point_in_log_block(const ivec2& block_start,
                                                                     const ivec2& out_extents,
                                                                     const Conv& conv,
                                                                     const Cb&& cb) {
  using BlockSetupT = typename Conv::BlockSetupT;
  using StaticConfigT = typename BlockSetupT::StaticConfigT;
  static_assert(Conv::lanes_axis == 1);
  const auto& block_dim = conv.block_setup.block_dim();
  auto coords = block_start + conv.block_setup.thread_idx();
  if (coords.x < out_extents.x) {
#pragma unroll
    for (int lane = 0; lane < StaticConfigT::lanes; lane++) {
      if (coords.y < out_extents.y) {
        cb(coords, lane);
        coords.y += block_dim.y;
      }
    }
  }
}

template <typename Conv, typename Cb>
DALI_DEVICE DALI_FORCEINLINE void for_each_output_point_in_log_block(const ivec3& block_start,
                                                                     const ivec3& out_extents,
                                                                     const Conv& conv,
                                                                     const Cb&& cb) {
  using BlockSetupT = typename Conv::BlockSetupT;
  using StaticConfigT = typename BlockSetupT::StaticConfigT;
  constexpr int lanes_axis = Conv::lanes_axis;
  static_assert(lanes_axis == 1 || lanes_axis == 2);
  constexpr int non_lanes_axis = Conv::lanes_axis == 2 ? 1 : 2;
  const auto& block_dim = conv.block_setup.block_dim();
  auto coords = block_start + conv.block_setup.thread_idx();
  if (coords.x < out_extents.x && coords[non_lanes_axis] < out_extents[non_lanes_axis]) {
#pragma unroll
    for (int lane = 0; lane < StaticConfigT::lanes; lane++) {
      if (coords[lanes_axis] < out_extents[lanes_axis]) {
        cb(coords, lane);
        coords[lanes_axis] += block_dim[lanes_axis];
      }
    }
  }
}

template <typename DoConv, typename In, typename Out>
DALI_DEVICE DALI_FORCEINLINE void stride_grid(const ivec2& block_start, const ivec2& grid_extents,
                                              const ivec2& out_extents, const In* __restrict__ in,
                                              Out* __restrict__ out, DoConv&& do_conv) {
  for (int y_start = block_start.y; y_start < out_extents.y; y_start += grid_extents.y) {
    for (int x_start = block_start.x; x_start < out_extents.x; x_start += grid_extents.x) {
      do_conv(ivec2{x_start, y_start}, in, out);
    }
  }
}

template <typename DoConv, typename In, typename Out>
DALI_DEVICE DALI_FORCEINLINE void stride_grid(const ivec3& block_start, const ivec3& grid_extents,
                                              const ivec3& out_extents, const In* __restrict__ in,
                                              Out* __restrict__ out, DoConv&& do_conv) {
  for (int z_start = block_start.z; z_start < out_extents.z; z_start += grid_extents.z) {
    for (int y_start = block_start.y; y_start < out_extents.y; y_start += grid_extents.y) {
      for (int x_start = block_start.x; x_start < out_extents.x; x_start += grid_extents.x) {
        do_conv(ivec3{x_start, y_start, z_start}, in, out);
      }
    }
  }
}

template <typename Conv, int axes>
DALI_DEVICE DALI_FORCEINLINE void stride_grid(const ivec<axes>& initial_block_start,
                                              const ivec<axes>& grid_extents,
                                              const ivec<axes>& out_extents,
                                              const i64vec<axes>& out_strides,
                                              const int64_t& out_frame_stride, const Conv& conv) {
  using BlockSetupT = typename Conv::BlockSetupT;
  using StaticConfigT = typename BlockSetupT::StaticConfigT;
  using SampleDescT = typename Conv::SampleDescT;
  using Acc = typename SampleDescT::Acc;
  using Out = typename SampleDescT::Out;
  constexpr int lanes = StaticConfigT::lanes;
  const auto* in = conv.sample_desc.in;
  auto* out = conv.sample_desc.out;
  for (int f = 0; f < conv.sample_desc.in_shape.num_frames;
       f++, in += conv.sample_desc.in_shape.frame_stride, out += out_frame_stride) {
    stride_grid(
        initial_block_start, grid_extents, out_extents, in, out,
        [&](const ivec<axes>& block_start, const auto* __restrict__ in, auto* __restrict__ out) {
          Acc acc[lanes]{};
          conv.compute(acc, in, block_start);
          for_each_output_point_in_log_block(
              block_start, out_extents, conv, [&](const auto& coords, int lane) {
                out[dot(coords, out_strides)] = ConvertSat<Out>(acc[lane]);
              });
        });
  }
}

/*
Given a HWC image and RS filter, all the necessary products for computing the convolution
explicitly can be seen as multiplying a matrix of shape HWC x RS with a vector of size RS. Now,
assuming a standard contiguious memory layout of the HWC image, if you look at any given column of
the HWC x RS matrix, consecutive rows map to consecutive memory addresses (with the exception of
positions when we cross H, W extents and border remapping takes place). This implementation melds
the W and C extents to utilize this property. Additionally, the implementation that uses shared
memory loads the inputs in blocks with extents corresponding to (D, )H, W * C extents to account
for the fact that spatailly close products will reuse some of the inputs.
*/
template <typename SampleDescT, typename BlockSetupProvider, typename InLoaderProvider,
          typename GridSetupT, typename OutShapeProviderT, typename ConvFactoryT>
__global__ void filter(const SampleDescT* __restrict__ descs,
                       BlockSetupProvider block_setup_provider, InLoaderProvider in_loader_provider,
                       GridSetupT grid_setup, OutShapeProviderT out_shape_provider,
                       ConvFactoryT conv_factory) {
  extern __shared__ char shm[];
  int sample_idx = grid_setup.sample_idx();
  for (int sample_idx = grid_setup.sample_idx(); sample_idx < grid_setup.num_samples();
       sample_idx += grid_setup.sample_dim()) {
    auto sample_desc = descs[sample_idx];
    const auto& block_setup = block_setup_provider(sample_idx);
    const auto& logical_block_extents = sample_desc.logical_block_extents;
    auto block_start = grid_setup.block_idx() * logical_block_extents;
    auto out_shape = out_shape_provider(sample_desc);
    // If extents are equal then strides are too, make it explicit as it faster and the usual case
    // (roi is, for now, only used for ``valid`` mode)
    if (any_coord(block_start >= out_shape.extents)) {
      continue;  // early exit to avoid all the setup only to do nothing in the stride loop
    }
    auto grid_size = grid_setup.grid_dim() * logical_block_extents;
    const auto& in_loader = in_loader_provider(sample_idx);
    with_conv(conv_factory, sample_desc, in_loader, block_setup, shm, [&](auto&& conv) {
      stride_grid(block_start, grid_size, out_shape.extents, out_shape.strides,
                  out_shape.frame_stride, conv);
    });
  }
}

/*
* The ``lanes`` paramter impacts perf in number of ways:
* 1. It reduces overhead of the for loops arithmetic when iterating over the filter extents:
*    We do not know the filter extents in compile time so those loops cannot be unrolled.
     However, using ``lanes`` we can handle multiple (i.e. exactly ``lanes``) outputs
     in a single pass of the filter loops, which amortises the total overhead.
  2. It increases the logical block size and workspace size in the shm variant,
     which allows for reusing of input values that lie close to each other (i.e., in given extent
     the distance is less than the corresponding extent of the filter).
  3. It also can deteriorate the perf: point 1 by increasing the registers
     pressure, point 2 by increasing the shm consumption.

For those reasons, the volumetric variant chooses the lanes extent (between y and z)
depending on the filter shape. On the one hand, the register pressure becomes too big
if we use lanes for both z and y extent. On the other, if the kernel is used for
separable convolution, it is important that the lanes extent matches non-trivial
filter extent, so that the input values reusing takes place.
*/
template <int axes>
struct StaticConfig {};

template <>
struct StaticConfig<2> {
  static constexpr int axes = 2;
  static constexpr int threadblock_size = 128;
  static constexpr int lanes = 16;
  static constexpr int max_grid_extent = 32;
  static constexpr int max_num_samples = 32;

  static DALI_HOST_DEV ivec2 max_grid_extents() {
    return {max_grid_extent, max_grid_extent};
  }
};

template <>
struct StaticConfig<3> {
  static constexpr int axes = 3;
  static constexpr int threadblock_size = 128;
  static constexpr int lanes = 16;
  static constexpr int max_grid_extent = 16;
  static constexpr int max_num_samples = 32;

  static DALI_HOST_DEV ivec3 max_grid_extents() {
    return {max_grid_extent, max_grid_extent, max_grid_extent};
  }
};

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

    DALI_DEVICE DALI_FORCEINLINE int flat_idx() const {
      return threadIdx.x;
    }

    DALI_DEVICE DALI_FORCEINLINE ivec<axes> thread_idx() const {
      assert(threadIdx.y == 0);
      assert(threadIdx.z == 0);
      return (static_cast<int>(threadIdx.x) >> strides_log2) & (block_dim() - 1);
    }
  };

  DALI_DEVICE DALI_FORCEINLINE const BlockSetup& operator()(int sample_idx) {
    return blocks[sample_idx];
  }

  const BlockSetup* blocks;
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
    auto max_extents = StaticConfigT::max_grid_extents();
    return {blockIdx.x, blockIdx.y & (max_extents.y - 1), blockIdx.y / max_extents.y};
  }

  template <int axes_ = axes>
  std::enable_if_t<axes_ == 2, dim3> kernel_setup() const {
    auto grid_dim = this->grid_dim();
    auto grid_sample_extent = std::min(num_samples(), sample_dim());
    return {static_cast<unsigned int>(grid_dim.x), static_cast<unsigned int>(grid_dim.y),
            static_cast<unsigned int>(grid_sample_extent)};
  }

  template <int axes_ = axes>
  std::enable_if_t<axes_ == 3, dim3> kernel_setup() const {
    auto grid_dim = this->grid_dim();
    auto max_extents = StaticConfigT::max_grid_extents();
    auto grid_sample_extent = std::min(num_samples(), sample_dim());
    return {static_cast<unsigned int>(grid_dim.x),
            static_cast<unsigned int>(grid_dim.z * max_extents.y + grid_dim.y),
            static_cast<unsigned int>(grid_sample_extent)};
  }

  DALI_DEVICE int sample_idx() const {
    return blockIdx.z;
  }

  DALI_HOST_DEV int num_samples() const {
    return num_samples_;
  }

  DALI_HOST_DEV int sample_dim() const {
    return StaticConfigT::max_num_samples;
  }

  ivec<axes> num_blocks_;
  int num_samples_;
};

}  // namespace filter

template <typename Out, typename In, typename W, bool has_channel_dim, bool has_sequence_dim,
          int axes_, bool enable_roi_>
struct FilterGpu {
  /* It computes a correlation of the input and the filter.
  Flip filter in both dimensions for a convolution. */

  static constexpr int axes = axes_;
  static constexpr bool enable_roi = enable_roi_;
  static constexpr int ndim =
      static_cast<int>(has_sequence_dim) + axes + static_cast<int>(has_channel_dim);
  static constexpr int sequence_dim = has_sequence_dim ? 0 : -1;
  static constexpr int channels_dim = has_channel_dim ? ndim - 1 : -1;
  using Intermediate = decltype(std::declval<W>() * std::declval<In>());
  using StaticConfigT = filter::StaticConfig<axes>;
  using BlockSetupProviderT = filter::AdaptiveBlock<StaticConfigT>;
  using BlockSetupT = typename BlockSetupProviderT::BlockSetup;
  using StaticBlockFactoryT = filter::StaticBlock<StaticConfigT>;
  using StaticBlockT = typename StaticBlockFactoryT::BlockSetup;
  using GridSetupT = filter::GridSetup<StaticConfigT>;
  using InShapeDescT = filter::InShapeDesc<axes>;
  using OutShapeDescT = filter::OutShapeDesc<axes>;
  using WorkspaceDescT = filter::WorkspaceDesc<axes>;
  using SampleDescT = filter::SampleDesc<Out, In, W, Intermediate, axes>;

  void Run(KernelContext& ctx, const TensorListView<StorageGPU, Out, ndim>& out,
           const TensorListView<StorageGPU, const In, ndim>& in,
           const TensorListView<StorageGPU, const W, axes>& filters,
           const span<const ivec<axes>> anchors, boundary::BoundaryType border_type,
           const TensorListView<StorageGPU, const In, 0>& fill_values = {}) {
    auto num_samples = in.shape.num_samples();
    assert(out.num_samples() == num_samples && filters.num_samples() == num_samples &&
           anchors.size() == num_samples);
    assert(fill_values.num_samples() == num_samples || fill_values.num_samples() == 0);

    SetupSampleDescs(out, in, filters, anchors);
    SampleDescT* samples_desc_dev;
    std::tie(samples_desc_dev) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, samples_desc_);
    auto grid_setup = PrepareGridSetup(make_cspan(samples_desc_));
    WithBlockSetupProvider(ctx, [&](auto&& block_setup_provider) {
      WithInLoaderProvider(ctx, border_type, fill_values, [&](auto&& in_loader_provider) {
        WithOutShapeProvider([&](auto&& out_shape_provider) {
          WithConvFactory([&](auto&& conv_factory) {
            filter::filter<<<grid_setup.kernel_setup(), StaticConfigT::threadblock_size,
                             required_shm_size_, ctx.gpu.stream>>>(
                samples_desc_dev, block_setup_provider, in_loader_provider, grid_setup,
                out_shape_provider, conv_factory);
            CUDA_CALL(cudaGetLastError());
          });
        });
      });
    });
  }

 protected:
  template <typename T>
  vec<axes, T> ShapeAsVec(const TensorShape<ndim>& shape) {
    return shape2vec<axes, T>(skip_dim<sequence_dim>(skip_dim<channels_dim>(shape)));
  }

  template <typename Inshapes, typename FilterShapes>
  void ValidateNumericLimits(const Inshapes& in_shapes, const FilterShapes& filter_shapes) {
    const i64vec<axes> max_grid_logical_extents =
        static_cast<i64vec<axes>>(StaticConfigT::max_grid_extents() * StaticConfigT::lanes);
    // so that we can safely use grid extents as a stride in a for loop
    const i64vec<axes> max_sample_extents =
        std::numeric_limits<int>::max() - max_grid_logical_extents + 1;
    for (int sample_idx = 0; sample_idx < in_shapes.num_samples(); sample_idx++) {
      const auto& in_shape = in_shapes[sample_idx];
      int64_t num_frames = has_sequence_dim ? in_shape[sequence_dim] : 1;
      DALI_ENFORCE(num_frames <= std::numeric_limits<int>::max(),
                   make_string("The number of frames for sample of idx ", sample_idx,
                               " exceeds the limit of ", std::numeric_limits<int>::max(), ". Got ",
                               num_frames, " frames."));
      int64_t num_channels = has_channel_dim ? in_shape[channels_dim] : 1;
      i64vec<axes> in_extents = ShapeAsVec<int64_t>(in_shape);
      in_extents[0] *= num_channels;
      for (int dim = 0; dim < axes; dim++) {
        static_assert(axes <= 3);
        const std::array<std::string, 3> extent_names = {
            "combined width and the number of channels", "height", "depth"};
        DALI_ENFORCE(
            in_extents[dim] <= max_sample_extents[dim],
            make_string("The ", extent_names[dim], " of sample at index ", sample_idx, " is ",
                        in_extents[dim], ", which exceeds the supported limit of ",
                        max_sample_extents[dim], ". The sample's shape is: ", in_shape, "."));
      }
      const auto& filter_shape = filter_shapes[sample_idx];
      DALI_ENFORCE(volume(filter_shape) <= std::numeric_limits<int>::max(),
                   make_string("Volume of filter for sample of idx ", sample_idx, " is ",
                               volume(filter_shape), ", which exceeds the limit of ",
                               std::numeric_limits<int>::max(), ". The filter's shape is ",
                               filter_shape, "."));
    }
  }

  void SetupSampleDescs(const TensorListView<StorageGPU, Out, ndim>& out,
                        const TensorListView<StorageGPU, const In, ndim>& in,
                        const TensorListView<StorageGPU, const W, axes>& filters,
                        const span<const ivec<axes>> anchors) {
    const auto& out_shapes = out.shape;
    const auto& in_shapes = in.shape;
    const auto& filter_shapes = filters.shape;
    int num_samples = in_shapes.num_samples();
    ValidateNumericLimits(in_shapes, filter_shapes);
    samples_desc_.clear();
    samples_desc_.reserve(num_samples);
    block_setups_.clear();
    block_setups_.reserve(num_samples);
    all_fit_in_shm_workspace_ = true;
    required_shm_size_ = 0;
    const int shared_mem_limit = GetSharedMemPerBlock();
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      const auto& in_shape = in_shapes[sample_idx];
      int num_frames = has_sequence_dim ? in_shape[sequence_dim] : 1;
      int num_channels = has_channel_dim ? in_shape[channels_dim] : 1;
      ivec<axes> in_extents = ShapeAsVec<int>(in_shape);
      auto filter_extents = shape2vec(filter_shapes[sample_idx]);
      BlockSetupT block_setup = PrepareBlockSetup(in_extents, num_channels);
      int lanes_axis = GetLanesAxis(filter_extents);
      ivec<axes> logical_block_extents = block_setup.block_dim();
      logical_block_extents[lanes_axis] *= StaticConfigT::lanes;
      WorkspaceDescT workspace_desc;
      int required_workspace;
      SetupWorkspaceDesc(workspace_desc, required_workspace, filter_extents, logical_block_extents,
                         num_channels, shared_mem_limit);
      InShapeDescT shape_desc = PrepareSampleShapeDesc(
          in_extents, filter_extents, anchors[sample_idx], num_frames, num_channels);
      ivec<axes> out_extents = ShapeAsVec<int>(out_shapes[sample_idx]);
      OutShapeDescT out_shape_desc = PrepareOutputShapeDesc(out_extents, num_channels);
      required_shm_size_ = std::max(required_shm_size_, required_workspace);
      all_fit_in_shm_workspace_ &= required_workspace > 0;
      block_setups_.push_back(block_setup);
      samples_desc_.push_back({out.tensor_data(sample_idx), in.tensor_data(sample_idx),
                               filters.tensor_data(sample_idx), shape_desc, out_shape_desc,
                               workspace_desc, logical_block_extents, lanes_axis});
    }
  }

  InShapeDescT PrepareSampleShapeDesc(ivec<axes> in_extents, ivec<axes> filter_extents,
                                      ivec<axes> anchor_shift, int num_frames, int num_channels) {
    int width = in_extents.x;
    in_extents.x *= num_channels;
    int64_t frame_stride;
    i64vec<axes> in_strides;
    strides(in_strides, frame_stride, in_extents);
    ivec<axes> filter_strides;
    strides(filter_strides, filter_extents);
    anchor_shift.x *= num_channels;
    return {frame_stride, in_strides,     num_frames,
            width,        num_channels,   filter_extents.x * num_channels,
            in_extents,   filter_extents, filter_strides,
            anchor_shift};
  }

  void SetupWorkspaceDesc(WorkspaceDescT& workspace_desc, int& required_workspace,
                          ivec<axes> filter_extents, ivec<axes> logical_block_extents,
                          int num_channels, int shared_mem_limit) {
    i64vec<axes> in_workspace_extents = static_cast<i64vec<axes>>(filter_extents) - 1;
    in_workspace_extents.x *= num_channels;
    in_workspace_extents += logical_block_extents;
    int64_t in_workspace_num_elements = volume(in_workspace_extents);
    int64_t idx_workspace_size =
        std::accumulate(in_workspace_extents.begin() + 1, in_workspace_extents.end(), 0);
    int64_t in_workspace_offset = align_up(idx_workspace_size * sizeof(int), sizeof(In));
    int64_t workspace_size = in_workspace_offset + in_workspace_num_elements * sizeof(In);
    if (num_channels > logical_block_extents.x || workspace_size > shared_mem_limit) {
      in_workspace_extents = workspace_size = in_workspace_offset = 0;
    }
    required_workspace = workspace_size;
    workspace_desc.in_offset = in_workspace_offset;
    workspace_desc.in_extents = in_workspace_extents;
    strides(workspace_desc.in_strides, workspace_desc.in_extents);
  }

  OutShapeDescT PrepareOutputShapeDesc(ivec<axes> out_extents, int num_channels) {
    out_extents.x *= num_channels;
    int64_t out_frame_stride;
    i64vec<axes> out_strides;
    strides(out_strides, out_frame_stride, out_extents);
    return {out_frame_stride, out_strides, out_extents};
  }

  BlockSetupT PrepareBlockSetup(ivec2 in_extents, int num_channels) {
    in_extents.x *= num_channels;
    int total_block_log2 = dali::ilog2(StaticConfigT::threadblock_size);
    int sample_x_log2 = in_extents.x == 0 ? 0 : dali::ilog2(in_extents.x - 1) + 1;
    int block_x_log2 = std::min(total_block_log2, sample_x_log2);
    ivec2 block{block_x_log2, total_block_log2 - block_x_log2};
    return {block, {0, block.x}};
  }

  BlockSetupT PrepareBlockSetup(ivec3 in_extents, int num_channels) {
    in_extents.x *= num_channels;
    int total_block_log2 = dali::ilog2(StaticConfigT::threadblock_size);
    int sample_x_log2 = in_extents.x == 0 ? 0 : dali::ilog2(in_extents.x - 1) + 1;
    int sample_y_log2 = in_extents.y == 0 ? 0 : dali::ilog2(in_extents.y - 1) + 1;
    int block_x_log2 = std::min(sample_x_log2, total_block_log2);
    int block_y_log2 = std::min(sample_y_log2, total_block_log2 - block_x_log2);
    int block_z_log2 = total_block_log2 - block_x_log2 - block_y_log2;
    ivec3 block{block_x_log2, block_y_log2, block_z_log2};
    return {block, {0, block.x, block.x + block.y}};
  }

  GridSetupT PrepareGridSetup(span<const SampleDescT> sample_descs) {
    ivec<axes> num_blocks = 0;
    for (const auto& sample_desc : sample_descs) {
      auto sample_num_blocks =
          div_ceil(sample_desc.out_shape.extents, sample_desc.logical_block_extents);
      num_blocks = max(num_blocks, sample_num_blocks);
    }
    num_blocks = min(num_blocks, StaticConfigT::max_grid_extents());
    int num_samples = sample_descs.size();
    return {num_blocks, num_samples};
  }

  int GetLanesAxis(const ivec2 filter_extents) {
    (void)filter_extents;
    return 1;
  }

  int GetLanesAxis(const ivec3 filter_extents) {
    return filter_extents.y > filter_extents.z ? 1 : 2;
  }

  /**
   * @brief Selects block setup provider for kernel launch. Picks static block provider if possible
   * (which has less arithmetic overhead) or adaptive, per-sample provider if needed (i.e. when
   * samples have degenerated extents such as width = 1, which would result in very slow run with
   * static block).
   */
  template <typename KernelLauncher>
  void WithBlockSetupProvider(KernelContext& ctx, KernelLauncher&& launch_kernel) {
    if (std::all_of(block_setups_.begin(), block_setups_.end(), [](const auto& block_setup) {
          return block_setup.block_dim() == StaticBlockT{}.block_dim();
        })) {
      launch_kernel(StaticBlockFactoryT{});
    } else {
      BlockSetupT* block_setups_dev;
      std::tie(block_setups_dev) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, block_setups_);
      BlockSetupProviderT block_setup{block_setups_dev};
      launch_kernel(std::move(block_setup));
    }
  }

  /**
   * @brief Selected input loader provider for kernel launch that handles OOB accesess according to
   * selected border mode.
   */
  template <typename KernelLauncher>
  void WithInLoaderProvider(KernelContext& ctx, boundary::BoundaryType border_type,
                            const TensorListView<StorageGPU, const In, 0>& fill_values,
                            KernelLauncher&& launch_kernel) {
    using namespace boundary;  // NOLINT(build/namespaces)
    VALUE_SWITCH(border_type, BT, (
      BoundaryType::REFLECT_101, BoundaryType::REFLECT_1001,
      BoundaryType::CLAMP, BoundaryType::WRAP), (
        RunKernelBorderRemap<BT>(ctx, std::move(launch_kernel));
      ), (  // NOLINT
        if (border_type == BoundaryType::CONSTANT) {
          RunKernelBorderConstant(ctx, fill_values, std::move(launch_kernel));
        } else {
          DALI_FAIL(
            make_string("Unsupported border type was specified: ", to_string(border_type), "."));
        }
      ));  // NOLINT
  }

  template <boundary::BoundaryType border, typename KernelLauncher>
  void RunKernelBorderRemap(KernelContext& ctx, KernelLauncher&& launch_kernel) {
    using Loader = filter::InLoaderBorderRemap<In, axes, border>;
    filter::InLoaderProvider<Loader> loader_factory{};
    launch_kernel(std::move(loader_factory));
  }

  template <typename KernelLauncher>
  void RunKernelBorderConstant(KernelContext& ctx,
                               const TensorListView<StorageGPU, const In, 0>& fill_values,
                               KernelLauncher&& launch_kernel) {
    int num_samples = samples_desc_.size();
    if (fill_values.num_samples() != num_samples) {
      filter::InLoaderProvider<filter::InLoaderPad<In, axes>> loader_factory{nullptr};
      launch_kernel(std::move(loader_factory));
    } else {
      fill_values_.resize(num_samples);
      for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
        fill_values_[sample_idx] = fill_values[sample_idx].data;
      }
      const In** fill_values_dev;
      std::tie(fill_values_dev) = ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, fill_values_);
      filter::InLoaderProvider<filter::InLoaderPad<In, axes>> loader_factory{fill_values_dev};
      launch_kernel(std::move(loader_factory));
    }
  }

  /**
   * @brief If the ``enable_roi`` is set to false, the input and outpus are expected do have equal
   * shapes. Knowing that helps to spare a few registers when running the kernel.
   */
  template <typename KernelLauncher>
  void WithOutShapeProvider(KernelLauncher&& launch_kernel) {
    if (enable_roi) {
      launch_kernel(filter::OutShapeProviderROI<SampleDescT>{});
    } else {
      assert(std::all_of(samples_desc_.begin(), samples_desc_.end(), [](const auto& sample_desc) {
        return sample_desc.out_shape.extents == sample_desc.in_shape.in_extents;
      }));
      launch_kernel(filter::OutShapeProviderSame<SampleDescT>{});
    }
  }

  /**
   * @brief Runs the kernel with convolution that utilizes cuda shm if all samples
   * can fit there, otherwise chooses slower but more generic direct conv.
   */
  template <typename KernelLauncher>
  void WithConvFactory(KernelLauncher&& launch_kernel) {
    if (all_fit_in_shm_workspace_) {
      launch_kernel(filter::ShmConvFactory{});
    } else {
      launch_kernel(filter::DirectConvFactory{});
    }
  }

  std::vector<SampleDescT> samples_desc_;
  std::vector<const In*> fill_values_;
  std::vector<BlockSetupT> block_setups_;
  bool all_fit_in_shm_workspace_;
  int required_shm_size_;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_FILTER_GPU_CUH_
