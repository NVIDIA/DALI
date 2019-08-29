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

#include "dali/core/common.h"
#include "dali/core/geom/vec.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/imgproc/warp/mapping_traits.h"
#include "dali/kernels/imgproc/sampler.h"
#include "dali/kernels/imgproc/warp/map_coords.h"

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

  static_assert(spatial_ndim == 2, "Not implemented for spatial_ndim != 2");

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
  template <DALIInterpType static_interp>
  void RunImpl(
      KernelContext &context,
      const OutTensorCPU<OutputType, 3> &output,
      const InTensorCPU<InputType, 3> &input,
      Mapping &mapping,
      const BorderType &border = {}) {
    // 2D HWC implementation.
    // 3D will be added as an overload for input/output ndim == 4.
    int out_w = output.shape[1];
    int out_h = output.shape[0];
    int c     = output.shape[2];
    int in_w  = input.shape[1];
    int in_h  = input.shape[0];

    Surface2D<OutputType> out = {
      output.data,
      out_w, out_h, c,
      c, out_w*c, 1
    };
    Surface2D<const InputType> in = {
      input.data,
      in_w, in_h, c,
      c, in_w*c, 1
    };

    Sampler<static_interp, InputType> sampler(in);

    for (int y = 0; y < out_h; y++) {
      for (int x = 0; x < out_w; x++) {
        auto src = warp::map_coords(mapping, ivec2(x, y));
        sampler(&out(x, y), src, border);
      }
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_WARP_CPU_H_
