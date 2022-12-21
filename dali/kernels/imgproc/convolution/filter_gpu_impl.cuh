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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_FILTER_GPU_IMPL_CUH_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_FILTER_GPU_IMPL_CUH_

#include "dali/core/boundary.h"
#include "dali/core/convert.h"
#include "dali/core/geom/vec.h"

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
  ivec<axes> filter_strides;  // 1, s(, sr)
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
  // for precomputed indices. At that offset, the
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
  int lane_axis;
};

/** @defgroup InputLoader InputLoader is meant to specialize loading of the sample from global
 * memory by providing different means of handling OOB accesses.
 * @{
 */
template <typename In, int axes, BoundaryType border>
class InLoaderBorderRemap {
  template <BoundaryType border_>
  using BorderTag = std::integral_constant<BoundaryType, border_>;

 public:
  DALI_HOST_DEV DALI_FORCEINLINE int border_remap(int idx, InShapeDesc<axes> sample_shape,
                                                  int axis) const {
    return border_remap(idx, sample_shape.in_extents[axis], BorderTag<border>{});
  }

  /**
   * The innermost extent consists of the width and channel extents flattened.
   * Thus, handling border condition for innermost extent requires extra step of computing back
   * the current channel and spatial position.
   */
  DALI_HOST_DEV DALI_FORCEINLINE int border_remap_innermost(int idx,
                                                            InShapeDesc<axes> sample_shape) const {
    int num_channels = sample_shape.num_channels;
    if (idx < 0) {
      // First, shift by 1 towards 0 (idx + 1), so that indices belonging to the same pixel
      // translate to the same number pixel index when dividing by the channels stride
      // (-6, -5, -4, -3, -2, -1) + 1 -> (-5, -4, -3, -2, -1, 0)
      // (-5, -4, -3, -2, -1, 0) / 3 -> (-1, -1, -1, 0, 0, 0)
      // Then shift back away from 0 (-1, -1, -1, 0, 0, 0) - 1 -> (-2, -2, -2, -1, -1, -1)
      // Finally, with (num_channels - 1) we get the positive channel indices
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

  DALI_HOST_DEV DALI_FORCEINLINE In load(const In* __restrict__ in, ivec<axes> coords,
                                         InShapeDesc<axes> sample_shape) const {
    return in[dot(coords, sample_shape.in_strides)];
  }

 private:
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
};

template <typename In, int axes>
class InLoaderPad {
 public:
  DALI_HOST_DEV DALI_FORCEINLINE int border_remap(int idx, InShapeDesc<axes> sample_shape,
                                                  int axis) const {
    (void)sample_shape;
    (void)axis;
    return idx;
  }

  DALI_HOST_DEV DALI_FORCEINLINE int border_remap_innermost(int idx,
                                                            InShapeDesc<axes> sample_shape) const {
    (void)sample_shape;
    return idx;
  }

  DALI_HOST_DEV DALI_FORCEINLINE In load(const In* __restrict__ in, ivec<axes> coords,
                                         InShapeDesc<axes> sample_shape) const {
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
 * First loads the input patch necessary to compute the output of logical block size
 * (logical_block_extents) into shared memory (including filter's halo/apron), then computes the
 * convolution.
 */
template <int lane_axis_, typename SampleDescT_, typename Inloader_, typename BlockSetupT_>
class ShmInputConv {
 public:
  using SampleDescT = SampleDescT_;
  using Inloader = Inloader_;
  using BlockSetupT = BlockSetupT_;
  using StaticConfigT = typename BlockSetupT::StaticConfigT;
  using In = typename SampleDescT::In;
  using Acc = typename SampleDescT::Acc;
  static constexpr int lane_axis = lane_axis_;

  DALI_DEVICE DALI_FORCEINLINE ShmInputConv(SampleDescT sample_desc, Inloader in_loader,
                                            BlockSetupT block_setup, In* in_workspace,
                                            int* precomputed_idx)
      : sample_desc_{sample_desc},
        in_loader_{in_loader},
        block_setup_{block_setup},
        in_workspace_{in_workspace},
        precomputed_idx_{precomputed_idx} {}

  DALI_DEVICE DALI_FORCEINLINE void compute(Acc* acc, const In* __restrict__ in,
                                            ivec<SampleDescT::axes> start) const {
    auto anchored_start = start - sample_desc_.in_shape.anchor_shift;
    __syncthreads();
    precompute_indices(anchored_start);
    __syncthreads();
    load_input_to_shm(in, anchored_start);
    __syncthreads();
    auto thread_idx = block_setup_.thread_idx();
    compute_conv_lanes(
        sample_desc_.in_shape.filter_extents, sample_desc_.in_shape.in_filter_width,
        [&](auto filter_coef, auto offset, int lane) {
          auto in_val =
              in_workspace_[dot(thread_idx + offset, sample_desc_.workspace_desc.in_strides)];
          acc[lane] += in_val * filter_coef;
        });
  }

  DALI_DEVICE DALI_FORCEINLINE const SampleDescT& sample_desc() const {
    return sample_desc_;
  }

  DALI_DEVICE DALI_FORCEINLINE const BlockSetupT& block_setup() const {
    return block_setup_;
  }

 private:
  /**
   * @brief Iterates over all filter positions and produces offsets of the corresponding
   * positions in the input.
   *
   * For each filter position, `lanes` offsets are computed to amortize the overhead of
   * the for loops.
   */
  template <typename ProcessInputPosition>
  DALI_DEVICE DALI_FORCEINLINE void compute_conv_lanes(
      ivec2 filter_extents, int in_filter_width, ProcessInputPosition&& process_position) const {
    ivec2 block_dim = block_setup_.block_dim();
    const auto* filter = sample_desc_.filter;
    for (int r = 0; r < filter_extents.y; r++) {
      for (int s = 0; s < in_filter_width; s += sample_desc_.in_shape.num_channels) {
        auto filter_coef = __ldg(filter++);
#pragma unroll
        for (int lane = 0, lanes_offset = 0; lane < StaticConfigT::lanes;
             lane++, lanes_offset += block_dim.y) {
          ivec2 offset{s, r + lanes_offset};
          process_position(filter_coef, offset, lane);
        }
      }
    }
  }

  /**
   * @brief Iterates over all filter positions and produces offsets of the corresponding
   * positions in the input.
   *
   * For each filter position, `lanes` offsets are computed
   * to amortize the overhead of the for loops.
   */
  template <typename ProcessInputPosition>
  DALI_DEVICE DALI_FORCEINLINE void compute_conv_lanes(
      ivec3 filter_extents, int in_filter_width, ProcessInputPosition&& process_position) const {
    ivec3 block_dim = block_setup_.block_dim();
    const auto* filter = sample_desc_.filter;
    for (int p = 0; p < filter_extents.z; p++) {
      for (int r = 0; r < filter_extents.y; r++) {
        for (int s = 0; s < in_filter_width; s += sample_desc_.in_shape.num_channels) {
          auto filter_coef = __ldg(filter++);
          ivec3 filter_position{s, r, p};
#pragma unroll
          for (int lane = 0; lane < StaticConfigT::lanes; lane++) {
            process_position(filter_coef, filter_position, lane);
            filter_position[lane_axis] += block_dim[lane_axis];
          }
        }
      }
    }
  }

  DALI_DEVICE DALI_FORCEINLINE void precompute_indices(ivec2 anchored_start) const {
    for (int y = block_setup_.flat_idx(); y < sample_desc_.workspace_desc.in_extents.y;
         y += block_setup_.flat_size()) {
      precomputed_idx_[y] = in_loader_.border_remap(anchored_start.y + y, sample_desc_.in_shape, 1);
    }
  }

  DALI_DEVICE DALI_FORCEINLINE void precompute_indices(ivec3 anchored_start) const {
    int* ys = precomputed_idx_;
    for (int y = block_setup_.flat_idx(); y < sample_desc_.workspace_desc.in_extents.y;
         y += block_setup_.flat_size()) {
      ys[y] = in_loader_.border_remap(anchored_start.y + y, sample_desc_.in_shape, 1);
    }
    int* zs = precomputed_idx_ + sample_desc_.workspace_desc.in_extents.y;
    for (int z = block_setup_.flat_idx(); z < sample_desc_.workspace_desc.in_extents.z;
         z += block_setup_.flat_size()) {
      zs[z] = in_loader_.border_remap(anchored_start.z + z, sample_desc_.in_shape, 2);
    }
  }

  DALI_DEVICE DALI_FORCEINLINE void load_input_to_shm(const In* __restrict__ in,
                                                      ivec2 anchored_start) const {
    ivec2 block_dim = block_setup_.block_dim();
    ivec2 thread_idx = block_setup_.thread_idx();
    for (int x = thread_idx.x; x < sample_desc_.workspace_desc.in_extents.x; x += block_dim.x) {
      int global_x = in_loader_.border_remap_innermost(anchored_start.x + x, sample_desc_.in_shape);
      for (int y = thread_idx.y; y < sample_desc_.workspace_desc.in_extents.y; y += block_dim.y) {
        int global_y = precomputed_idx_[y];
        in_workspace_[dot(ivec2{x, y}, sample_desc_.workspace_desc.in_strides)] =
            in_loader_.load(in, ivec2{global_x, global_y}, sample_desc_.in_shape);
      }
    }
  }

  DALI_DEVICE DALI_FORCEINLINE void load_input_to_shm(const In* __restrict__ in,
                                                      ivec3 anchored_start) const {
    const int* const __restrict__ ys = precomputed_idx_;
    const int* const __restrict__ zs = precomputed_idx_ + sample_desc_.workspace_desc.in_extents.y;
    ivec3 block_dim = block_setup_.block_dim();
    ivec3 thread_idx = block_setup_.thread_idx();
    for (int x = thread_idx.x; x < sample_desc_.workspace_desc.in_extents.x; x += block_dim.x) {
      int global_x = in_loader_.border_remap_innermost(anchored_start.x + x, sample_desc_.in_shape);
      for (int y = thread_idx.y; y < sample_desc_.workspace_desc.in_extents.y; y += block_dim.y) {
        int global_y = ys[y];
        for (int z = thread_idx.z; z < sample_desc_.workspace_desc.in_extents.z; z += block_dim.z) {
          int global_z = zs[z];
          in_workspace_[dot(ivec3{x, y, z}, sample_desc_.workspace_desc.in_strides)] =
              in_loader_.load(in, ivec3{global_x, global_y, global_z}, sample_desc_.in_shape);
        }
      }
    }
  }

  SampleDescT sample_desc_;
  Inloader in_loader_;
  BlockSetupT block_setup_;
  In* in_workspace_;
  int* precomputed_idx_;
};

/**
 * @brief Computes the convolution of logical block size size (logical_block_extents).
 *
 * Accesses the input directly in global memory.
 * Used as a fallback when the filter size or number of channels in the input make it
 * impossible to use ``ShmInputConv``.
 */
template <int lane_axis_, typename SampleDescT_, typename Inloader_, typename BlockSetupT_>
class DirectInputConv {
 public:
  using SampleDescT = SampleDescT_;
  using Inloader = Inloader_;
  using BlockSetupT = BlockSetupT_;
  using StaticConfigT = typename BlockSetupT::StaticConfigT;
  using Acc = typename SampleDescT::Acc;
  using In = typename SampleDescT::In;
  static constexpr int lane_axis = lane_axis_;

  DALI_DEVICE DALI_FORCEINLINE DirectInputConv(SampleDescT sample_desc, Inloader in_loader,
                                               BlockSetupT block_setup)
      : sample_desc_{sample_desc}, in_loader_{in_loader}, block_setup_{block_setup} {}

  DALI_DEVICE DALI_FORCEINLINE void compute(Acc* acc, const In* __restrict__ in,
                                            ivec2 start) const {
    ivec2 block_dim = block_setup_.block_dim();
    ivec2 thread_idx = block_setup_.thread_idx();
    ivec2 coords = start - sample_desc_.in_shape.anchor_shift + thread_idx;
    const auto* filter = sample_desc_.filter;
    for (int s = 0; s < sample_desc_.in_shape.filter_extents.x; s++) {
      auto global_x = in_loader_.border_remap_innermost(
          coords.x + s * sample_desc_.in_shape.num_channels, sample_desc_.in_shape);
      for (int r = 0; r < sample_desc_.in_shape.filter_extents.y; r++) {
        auto filter_coef = __ldg(filter + dot(ivec2{s, r}, sample_desc_.in_shape.filter_strides));
        // Even without shm, using `lanes` speeds up the kernel by reducing
        // the cost of nested loops arithmetic per single output value
#pragma unroll
        for (int lane = 0; lane < StaticConfigT::lanes; lane++) {
          auto global_y =
              in_loader_.border_remap(coords.y + r + lane * block_dim.y, sample_desc_.in_shape, 1);
          auto in_val = in_loader_.load(in, ivec2{global_x, global_y}, sample_desc_.in_shape);
          acc[lane] += in_val * filter_coef;
        }
      }
    }
  }

  DALI_DEVICE DALI_FORCEINLINE void compute(Acc* acc, const In* __restrict__ in,
                                            ivec3 start) const {
    constexpr int non_lane_axis = lane_axis == 1 ? 2 : 1;
    ivec3 block_dim = block_setup_.block_dim();
    ivec3 thread_idx = block_setup_.thread_idx();
    ivec3 filter_extents = sample_desc_.in_shape.filter_extents;
    ivec3 in_coords = start - sample_desc_.in_shape.anchor_shift + thread_idx;
    const auto* filter = sample_desc_.filter;
    for (int s = 0; s < filter_extents.x; s++) {
      int global_x = in_loader_.border_remap_innermost(
          in_coords.x + s * sample_desc_.in_shape.num_channels, sample_desc_.in_shape);
      for (int i = 0; i < filter_extents[non_lane_axis]; i++) {
        int global_non_lane_axis = in_loader_.border_remap(in_coords[non_lane_axis] + i,
                                                           sample_desc_.in_shape, non_lane_axis);
        for (int j = 0; j < filter_extents[lane_axis]; j++) {
          ivec3 filter_pos{s, lane_axis == 1 ? j : i, lane_axis == 1 ? i : j};
          auto filter_coef = __ldg(filter + dot(filter_pos, sample_desc_.in_shape.filter_strides));
          int lane_axis_pos = in_coords[lane_axis] + j;
          // Even without shm, using `lanes` speeds up the kernel by reducing
          // the cost of nested loops arithmetic per single output value
#pragma unroll
          for (int lane = 0; lane < StaticConfigT::lanes; lane++) {
            int global_lane_axis = in_loader_.border_remap(
                lane_axis_pos + lane * block_dim[lane_axis], sample_desc_.in_shape, lane_axis);
            ivec3 coords{global_x, lane_axis == 1 ? global_lane_axis : global_non_lane_axis,
                         lane_axis == 1 ? global_non_lane_axis : global_lane_axis};
            auto in_val = in_loader_.load(in, coords, sample_desc_.in_shape);
            acc[lane] += in_val * filter_coef;
          }
        }
      }
    }
  }

  DALI_DEVICE DALI_FORCEINLINE const SampleDescT& sample_desc() const {
    return sample_desc_;
  }

  DALI_DEVICE DALI_FORCEINLINE const BlockSetupT& block_setup() const {
    return block_setup_;
  }

 private:
  SampleDescT sample_desc_;
  Inloader in_loader_;
  BlockSetupT block_setup_;
};


struct ShmConvFactory {
  template <int lane_axis, typename SampleDescT, typename Inloader, typename BlockSetupT>
  DALI_DEVICE DALI_FORCEINLINE ShmInputConv<lane_axis, SampleDescT, Inloader, BlockSetupT> create(
      SampleDescT sample_desc, Inloader in_loader, BlockSetupT block_setup, char* shm) const {
    using In = typename SampleDescT::In;
    int* idx_workspace = reinterpret_cast<int*>(shm);
    In* in_workspace = reinterpret_cast<In*>(shm + sample_desc.workspace_desc.in_offset);
    return {sample_desc, in_loader, block_setup, in_workspace, idx_workspace};
  }
};

struct DirectConvFactory {
  template <int lane_axis, typename SampleDescT, typename Inloader, typename BlockSetupT>
  DALI_DEVICE DALI_FORCEINLINE DirectInputConv<lane_axis, SampleDescT, Inloader, BlockSetupT>
  create(SampleDescT sample_desc, Inloader in_loader, BlockSetupT block_setup, char* shm) const {
    (void)shm;
    return {sample_desc, in_loader, block_setup};
  }
};

template <typename ConvFactoryT, typename SampleDescT, typename InLoaderT, typename BlockSetupT,
          typename Cb>
DALI_DEVICE DALI_FORCEINLINE std::enable_if_t<SampleDescT::axes == 2, void> with_conv(
    ConvFactoryT conv_factory, SampleDescT sample_desc, InLoaderT in_loader,
    BlockSetupT block_setup, char* shm, Cb&& cb) {
  cb(conv_factory.template create<1>(sample_desc, in_loader, block_setup, shm));
}

template <typename ConvFactoryT, typename SampleDescT, typename InLoaderT, typename BlockSetupT,
          typename Cb>
DALI_DEVICE DALI_FORCEINLINE std::enable_if_t<SampleDescT::axes == 3, void> with_conv(
    ConvFactoryT conv_factory, SampleDescT sample_desc, InLoaderT in_loader,
    BlockSetupT block_setup, char* shm, Cb&& cb) {
  if (sample_desc.lane_axis == 2) {
    cb(conv_factory.template create<2>(sample_desc, in_loader, block_setup, shm));
  } else {
    assert(sample_desc.lane_axis == 1);
    cb(conv_factory.template create<1>(sample_desc, in_loader, block_setup, shm));
  }
}

/** @} */  // end of InputConv

template <typename Conv, typename Cb>
DALI_DEVICE DALI_FORCEINLINE void for_each_output_point_in_log_block(ivec2 block_start,
                                                                     ivec2 out_extents,
                                                                     const Conv& conv, Cb&& cb) {
  using BlockSetupT = typename Conv::BlockSetupT;
  using StaticConfigT = typename BlockSetupT::StaticConfigT;
  static_assert(Conv::lane_axis == 1);
  ivec2 block_dim = conv.block_setup().block_dim();
  ivec2 coords = block_start + conv.block_setup().thread_idx();
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
DALI_DEVICE DALI_FORCEINLINE void for_each_output_point_in_log_block(ivec3 block_start,
                                                                     ivec3 out_extents,
                                                                     const Conv& conv, Cb&& cb) {
  using BlockSetupT = typename Conv::BlockSetupT;
  using StaticConfigT = typename BlockSetupT::StaticConfigT;
  constexpr int lane_axis = Conv::lane_axis;
  static_assert(lane_axis == 1 || lane_axis == 2);
  constexpr int non_lane_axis = Conv::lane_axis == 2 ? 1 : 2;
  ivec3 block_dim = conv.block_setup().block_dim();
  ivec3 coords = block_start + conv.block_setup().thread_idx();
  if (coords.x < out_extents.x && coords[non_lane_axis] < out_extents[non_lane_axis]) {
#pragma unroll
    for (int lane = 0; lane < StaticConfigT::lanes; lane++) {
      if (coords[lane_axis] < out_extents[lane_axis]) {
        cb(coords, lane);
        coords[lane_axis] += block_dim[lane_axis];
      }
    }
  }
}

template <typename DoConv, typename In, typename Out>
DALI_DEVICE DALI_FORCEINLINE void stride_grid(ivec2 block_start, ivec2 grid_extents,
                                              ivec2 out_extents, const In* __restrict__ in,
                                              Out* __restrict__ out, DoConv&& do_conv) {
  for (int y_start = block_start.y; y_start < out_extents.y; y_start += grid_extents.y) {
    for (int x_start = block_start.x; x_start < out_extents.x; x_start += grid_extents.x) {
      do_conv(ivec2{x_start, y_start}, in, out);
    }
  }
}

template <typename DoConv, typename In, typename Out>
DALI_DEVICE DALI_FORCEINLINE void stride_grid(ivec3 block_start, ivec3 grid_extents,
                                              ivec3 out_extents, const In* __restrict__ in,
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
DALI_DEVICE DALI_FORCEINLINE void stride_grid(ivec<axes> initial_block_start,
                                              ivec<axes> grid_extents, OutShapeDesc<axes> out_shape,
                                              const Conv& conv) {
  using BlockSetupT = typename Conv::BlockSetupT;
  using StaticConfigT = typename BlockSetupT::StaticConfigT;
  using SampleDescT = typename Conv::SampleDescT;
  using Acc = typename SampleDescT::Acc;
  using Out = typename SampleDescT::Out;
  constexpr int lanes = StaticConfigT::lanes;
  const auto* in = conv.sample_desc().in;
  auto* out = conv.sample_desc().out;
  for (int f = 0; f < conv.sample_desc().in_shape.num_frames;
       f++, in += conv.sample_desc().in_shape.frame_stride, out += out_shape.frame_stride) {
    stride_grid(initial_block_start, grid_extents, out_shape.extents, in, out,
                [&](ivec<axes> block_start, const auto* __restrict__ in, auto* __restrict__ out) {
                  Acc acc[lanes]{};
                  conv.compute(acc, in, block_start);
                  for_each_output_point_in_log_block(
                      block_start, out_shape.extents, conv, [&](ivec<axes> coords, int lane) {
                        out[dot(coords, out_shape.strides)] = ConvertSat<Out>(acc[lane]);
                      });
                });
  }
}

/**
 * Given a HWC image and RS filter, all the necessary products for computing the convolution
 * can be seen as multiplying a matrix of shape HWC x RS with a vector of size RS
 * (img2col). The kernel does not construct the matrix explicitly, but computes the convolution
 * in a similar manner - each threadblock computes a contigious parts of the output vector
 * by accumulating products of the filter and consecutive columns of the sub-array.
 * The output vector parts that are computed by a single threadblock have the size equal
 * to the logical threablock width, and there are ``logical threablock height`` of them
 * (or height and depth for 3D). In a usual case, it means that the output part has size equal
 * to the number of cuda threads and there are ``lanes`` parts).
 * Assuming a standard contiguious memory layout of the HWC image, if you look at any given column
 * of the HWC x RS matrix, the consecutive rows map to consecutive memory addresses
 * (with the exception of positions when we cross H, W extents and border remapping takes place).
 * This implementation melds the W and C extents to utilize this property.
 * Additionally, the implementation that uses shared memory loads the input to shm
 * in blocks with extents corresponding to (D, )H, W * C extents to account
 * for the fact that convolutions of spatially close outputs reuse some of the inputs.
 */
template <typename SampleDescT, typename BlockSetupProvider, typename InLoaderProvider,
          typename GridSetupT, typename OutShapeProviderT, typename ConvFactoryT>
__global__ void filter(const SampleDescT* __restrict__ descs,
                       BlockSetupProvider block_setup_provider, InLoaderProvider in_loader_provider,
                       GridSetupT grid_setup, OutShapeProviderT out_shape_provider,
                       ConvFactoryT conv_factory) {
  extern __shared__ char shm[];
  constexpr int axes = SampleDescT::axes;
  int sample_idx = grid_setup.sample_idx();
  for (int sample_idx = grid_setup.sample_idx(); sample_idx < grid_setup.num_samples();
       sample_idx += grid_setup.max_num_samples()) {
    auto sample_desc = descs[sample_idx];
    auto block_setup = block_setup_provider(sample_idx);
    ivec<axes> logical_block_extents = sample_desc.logical_block_extents;
    ivec<axes> block_start = grid_setup.block_idx() * logical_block_extents;
    auto out_shape = out_shape_provider(sample_desc);
    if (any_coord(block_start >= out_shape.extents)) {
      continue;  // early exit to avoid all the setup only to do nothing in the stride loop
    }
    auto grid_size = grid_setup.grid_dim() * logical_block_extents;
    const auto& in_loader = in_loader_provider(sample_idx);
    with_conv(conv_factory, sample_desc, in_loader, block_setup, shm,
              [&](auto&& conv) { stride_grid(block_start, grid_size, out_shape, conv); });
  }
}

/**
*
* The ``lanes`` parameter impacts perf in number of ways:
* 1. It reduces overhead of the for loops arithmetic when iterating over the filter extents:
*    We do not know the filter extents at compile time so those loops cannot be unrolled.
*    However, using ``lanes`` we can handle multiple (i.e. exactly ``lanes``) outputs
*    in a single pass of the filter loops, which amortises the total overhead.
* 2. It increases the logical block size and workspace size in the shm variant,
*    which allows for reusing of input values that lie close to each other (i.e., in given extent
*    the distance is less than the corresponding extent of the filter).
* 3. It also can deteriorate the perf: point 1 by increasing the registers
*    pressure, point 2 by increasing the shm consumption.

* For those reasons, the volumetric variant chooses the lanes extent (between y and z)
* depending on the filter shape. On the one hand, the register pressure becomes too big
* if we use lanes for both z and y extent. On the other, if the kernel is used for
* separable convolution, it is important that the lanes extent matches non-trivial
* filter extent, so that the input values reusing takes place.
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

/**
 * @brief Wrapper around threadIdx and blockDim.
 *
 * The StaticBlock Setup organizes threads in just a flat row of
 * threads - it seems to be the fastest layout due to less arithmetic and
 * contigious accesses to memory. However, for samples whose
 * width << threadblock_size, it may lead to poor performance.
 * For those cases there is AdaptiveBlock that reorganizes the threads accordingly
 * on per sample basis.
 */
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

/**
 * @brief Remaps a flat threablockIdx.x to x, y(, z) extents on per sample basis.
 * The extents are powers of two such that xyz = threadblock.x
 *
 * Those computations have some overhead, so the StaticBlock variant is preffered,
 * but for samples of degenerated shape (like very high samples with
 * width << threadblock size), the performance would be very poor
 * with the default static block.
 */
template <typename StaticConfigT_>
struct AdaptiveBlock {
  class BlockSetup {
   public:
    using StaticConfigT = StaticConfigT_;
    static constexpr int axes = StaticConfigT::axes;

    template <int axes_ = axes, std::enable_if_t<axes_ == 2, bool> = true>
    BlockSetup(ivec2 in_extents, int num_channels) {
      in_extents.x *= num_channels;
      int total_block_log2 = dali::ilog2(StaticConfigT::threadblock_size);
      int sample_x_log2 = in_extents.x == 0 ? 0 : dali::ilog2(in_extents.x - 1) + 1;
      extents_log2_.x = std::min(total_block_log2, sample_x_log2);
      extents_log2_.y = total_block_log2 - extents_log2_.x;
      strides_log2_ = {0, extents_log2_.x};
    }

    template <int axes_ = axes, std::enable_if_t<axes_ == 3, bool> = true>
    BlockSetup(ivec3 in_extents, int num_channels) {
      in_extents.x *= num_channels;
      int total_block_log2 = dali::ilog2(StaticConfigT::threadblock_size);
      int sample_x_log2 = in_extents.x == 0 ? 0 : dali::ilog2(in_extents.x - 1) + 1;
      int sample_y_log2 = in_extents.y == 0 ? 0 : dali::ilog2(in_extents.y - 1) + 1;
      extents_log2_.x = std::min(sample_x_log2, total_block_log2);
      extents_log2_.y = std::min(sample_y_log2, total_block_log2 - extents_log2_.x);
      extents_log2_.z = total_block_log2 - extents_log2_.x - extents_log2_.y;
      strides_log2_ = {0, extents_log2_.x, extents_log2_.x + extents_log2_.y};
    }

    DALI_DEVICE DALI_FORCEINLINE int flat_size() const {
      assert(blockDim.y == 1);
      assert(blockDim.z == 1);
      return blockDim.x;
    }

    DALI_HOST_DEV DALI_FORCEINLINE ivec<axes> block_dim() const {
      return 1 << extents_log2_;
    }

    DALI_DEVICE DALI_FORCEINLINE int flat_idx() const {
      return threadIdx.x;
    }

    DALI_DEVICE DALI_FORCEINLINE ivec<axes> thread_idx() const {
      assert(threadIdx.y == 0);
      assert(threadIdx.z == 0);
      return (static_cast<int>(threadIdx.x) >> strides_log2_) & (block_dim() - 1);
    }

   private:
    ivec<axes> extents_log2_;
    ivec<axes> strides_log2_;
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
    auto grid_sample_extent = std::min(num_samples(), max_num_samples());
    return {static_cast<unsigned int>(grid_dim.x), static_cast<unsigned int>(grid_dim.y),
            static_cast<unsigned int>(grid_sample_extent)};
  }

  template <int axes_ = axes>
  std::enable_if_t<axes_ == 3, dim3> kernel_setup() const {
    auto grid_dim = this->grid_dim();
    auto max_extents = StaticConfigT::max_grid_extents();
    auto grid_sample_extent = std::min(num_samples(), max_num_samples());
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

  DALI_HOST_DEV int max_num_samples() const {
    return StaticConfigT::max_num_samples;
  }

  ivec<axes> num_blocks_;
  int num_samples_;
};

}  // namespace filter
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_FILTER_GPU_IMPL_CUH_
