// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_WARP_CPU_H_
#define DALI_KERNELS_IMGPROC_WARP_CPU_H_

#include <algorithm>
#include "dali/core/common.h"
#include "dali/core/geom/vec.h"
#include "dali/core/geom/transform.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/imgproc/warp/mapping_traits.h"
#include "dali/kernels/imgproc/sampler.h"
#include "dali/kernels/imgproc/warp/map_coords.h"
#include "dali/kernels/imgproc/warp/affine.h"

namespace dali {
namespace kernels {

/**
 * @brief Performs generic warping of one tensor (on CPU)
 *
 * The warping uses a mapping functor to map destination coordinates to source
 * coordinates and samples the source tensor at the resulting locations.
 *
 * @remarks
 *  * Assumes HWC layout
 *  * Output and input have same number of spatial dimenions
 *  * Output and input have same number of channels and layout
 */
template <typename _Mapping, int _spatial_ndim, typename _OutputType, typename _InputType,
          typename _BorderType>
class WarpCPU {
 public:
  static constexpr int spatial_ndim = _spatial_ndim;
  static constexpr int tensor_ndim = spatial_ndim + 1;
  static constexpr int channel_dim = spatial_ndim;

  static_assert(spatial_ndim == 2 || spatial_ndim == 3, "WarpCPU only works for 2D and 3D");

  using Mapping = _Mapping;
  using OutputType = _OutputType;
  using InputType = _InputType;
  using BorderType = _BorderType;
  using MappingParams = warp::mapping_params_t<Mapping>;

  KernelRequirements Setup(
      KernelContext &context,
      const InTensorCPU<InputType, tensor_ndim> &input,
      const MappingParams &mapping_params,
      const TensorShape<spatial_ndim> &out_size,
      DALIInterpType interp = DALI_INTERP_LINEAR,
      const BorderType &border = {}) {
    KernelRequirements req;
    auto out_shape = shape_cat(out_size, input.shape[channel_dim]);
    req.output_shapes = { TensorListShape<tensor_ndim>({out_shape}) };
    return req;
  }

  void Run(
      KernelContext &context,
      const OutTensorCPU<OutputType, tensor_ndim> &output,
      const InTensorCPU<InputType, tensor_ndim> &input,
      const MappingParams &mapping_params,
      const TensorShape<spatial_ndim> &out_size,
      DALIInterpType interp = DALI_INTERP_LINEAR,
      const BorderType &border = {}) {
    Mapping mapping(mapping_params);

    assert(output.shape == shape_cat(out_size, input.shape[channel_dim]));

    VALUE_SWITCH(interp, static_interp, (DALI_INTERP_NN, DALI_INTERP_LINEAR),
      (RunImpl<static_interp>(context, output, input, mapping, border);),
      (DALI_FAIL("Unsupported interpolation type"))
    ); // NOLINT
  }

 private:
  template <DALIInterpType static_interp, typename Mapping_>
  void RunImpl(
      KernelContext &context,
      const OutTensorCPU<OutputType, 3> &output,
      const InTensorCPU<InputType, 3> &input,
      Mapping_ &mapping,
      BorderType border = {}) {
    int out_w = output.shape[1];
    int out_h = output.shape[0];
    int c     = output.shape[2];

    Surface2D<const InputType> in = as_surface_channel_last(input);

    Sampler2D<static_interp, InputType> sampler(in);

    for (int y = 0; y < out_h; y++) {
      OutputType *out_row = output(y, 0);
      for (int x = 0; x < out_w; x++) {
        auto src = warp::map_coords(mapping, ivec2(x, y));
        sampler(&out_row[c*x], src, border);
      }
    }
  }

  template <DALIInterpType static_interp, typename Mapping_>
  void RunImpl(
      KernelContext &context,
      const OutTensorCPU<OutputType, 4> &output,
      const InTensorCPU<InputType, 4> &input,
      Mapping_ &mapping,
      BorderType border = {}) {
    int out_w = output.shape[2];
    int out_h = output.shape[1];
    int out_d = output.shape[0];
    int c     = output.shape[3];

    Surface2D<const InputType> in = as_surface_channel_last(input);

    Sampler2D<static_interp, InputType> sampler(in);

    for (int z = 0; z < out_d; z++) {
      for (int y = 0; y < out_h; y++) {
        OutputType *out_row = output(z, y, 0);
        for (int x = 0; x < out_w; x++) {
          auto src = warp::map_coords(mapping, ivec3(x, y, z));
          sampler(&out_row[c*x], src, border);
        }
      }
    }
  }


  template <DALIInterpType static_interp>
  void RunImpl(
      KernelContext &context,
      const OutTensorCPU<OutputType, 3> &output,
      const InTensorCPU<InputType, 3> &input,
      AffineMapping<2> &mapping,
      BorderType border = {}) {
    int out_w = output.shape[1];
    int out_h = output.shape[0];
    int c     = output.shape[2];

    Surface2D<const InputType> in = as_surface_channel_last(input);

    Sampler2D<static_interp, InputType> sampler(in);

    // Optimization: instead of naively calculating source coordinates for each destination pixel,
    // we can exploit the linearity of the affine transform and just add ds/dx derivative
    // for each x.
    vec2 dsdx = mapping.transform.col(0);

    // use tiles to produce "checkpoints" to avoid excessive accumulation error
    constexpr int tile_w = 256;
    vec2 dsdx_tile = tile_w * dsdx;

    for (int y = 0; y < out_h; y++) {
      OutputType *out_row = output(y, 0);
      auto src_tile = warp::map_coords(mapping, ivec2(0, y));
      for (int x_tile = 0; x_tile < out_w; x_tile += tile_w, src_tile += dsdx_tile) {
        int x_tile_end = std::min(x_tile + tile_w, out_w);
        auto src = src_tile;
        for (int x = x_tile; x < x_tile_end; x++, src += dsdx) {
          sampler(&out_row[c*x], src, border);
        }
      }
    }
  }


  template <DALIInterpType static_interp>
  void RunImpl(
      KernelContext &context,
      const OutTensorCPU<OutputType, 4> &output,
      const InTensorCPU<InputType, 4> &input,
      AffineMapping<3> &mapping,
      BorderType border = {}) {
    int out_w = output.shape[2];
    int out_h = output.shape[1];
    int out_d = output.shape[0];
    int c     = output.shape[3];

    Surface3D<const InputType> in = as_surface_channel_last(input);

    Sampler3D<static_interp, InputType> sampler(in);

    // Optimization: instead of naively calculating source coordinates for each destination pixel,
    // we can exploit the linearity of the affine transform and just add ds/dx derivative
    // for each x.
    vec3 dsdx = mapping.transform.col(0);

    // use tiles to produce "checkpoints" to avoid excessive accumulation error
    constexpr int tile_w = 256;
    vec3 dsdx_tile = tile_w * dsdx;

    for (int z = 0; z < out_d; z++) {
      for (int y = 0; y < out_h; y++) {
        OutputType *out_row = output(z, y, 0);
        auto src_tile = warp::map_coords(mapping, ivec3(0, y, z));
        for (int x_tile = 0; x_tile < out_w; x_tile += tile_w, src_tile += dsdx_tile) {
          int x_tile_end = std::min(x_tile + tile_w, out_w);
          auto src = src_tile;
          for (int x = x_tile; x < x_tile_end; x++, src += dsdx) {
            sampler(&out_row[c*x], src, border);
          }
        }
      }
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_WARP_CPU_H_
