// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_WARP_WARP_SETUP_CUH_
#define DALI_KERNELS_IMGPROC_WARP_WARP_SETUP_CUH_

#include <vector>
#include <utility>
#include "dali/kernels/kernel.h"
#include "dali/kernels/common/block_setup.h"
#include "dali/core/geom/vec.h"

namespace dali {
namespace kernels {

/** @brief Contains implementation of warping kernels */
namespace warp {

template <int spatial_ndim, typename OutputType, typename InputType>
struct SampleDesc {
  OutputType *__restrict__ output;
  const InputType *__restrict__ input;
  ivec<spatial_ndim> out_size, in_size;
  i64vec<spatial_ndim> out_strides, in_strides;
  int channels;
  DALIInterpType interp;
};

/**
 * @brief Prepares batched execution of warping kernel.
 *
 * This class is a helper that calculates block setup and sample descriptors
 * for warping kernels. It's independent of actual mapping used.
 */
template <int spatial_ndim, typename OutputType, typename InputType>
class WarpSetup : public BlockSetup<spatial_ndim, spatial_ndim> {
  static_assert(spatial_ndim == 2 || spatial_ndim == 3,
    "Warping is defined only for 2D and 3D data with interleaved channels");

 public:
  using Base = BlockSetup<spatial_ndim, spatial_ndim>;
  using Base::tensor_ndim;
  using Base::Blocks;
  using Base::IsUniformSize;
  using Base::SetupBlocks;
  using Base::shape2size;
  using SampleDesc = warp::SampleDesc<spatial_ndim, OutputType, InputType>;
  using BlockDesc = kernels::BlockDesc<spatial_ndim>;

  KernelRequirements Setup(const TensorListShape<tensor_ndim> &output_shape,
                           bool force_variable_size = false) {
    SetupBlocks(output_shape, force_variable_size);

    KernelRequirements req = {};
    req.output_shapes = { output_shape };
    return req;
  }

  template <typename Backend>
  void PrepareSamples(const OutList<Backend, OutputType, tensor_ndim> &out,
                      const InList<Backend, InputType, tensor_ndim> &in,
                      span<const DALIInterpType> interp) {
    assert(out.num_samples() == in.num_samples());
    assert(interp.size() == in.num_samples() || interp.size() == 1);
    samples_.resize(in.num_samples());
    for (int i = 0; i < in.num_samples(); i++) {
      SampleDesc &sample = samples_[i];
      sample.input = in.tensor_data(i);
      sample.output = out.tensor_data(i);
      auto out_shape = out.tensor_shape(i);
      auto in_shape = in.tensor_shape(i);
      int channels = out_shape[spatial_ndim];
      sample.channels = channels;
      sample.out_size = shape2size(out_shape);
      sample.in_size = shape2size(in_shape);

      sample.out_strides.x = channels;
      sample.in_strides.x = channels;
      for (int d = 0; d < spatial_ndim - 1; d++) {
        sample.out_strides[d + 1] = sample.out_size[d] * sample.out_strides[d];
        sample.in_strides[d + 1]  = sample.in_size[d]  * sample.in_strides[d];
      }

      sample.interp = interp[interp.size() == 1 ? 0 : i];
    }
  }

  span<const SampleDesc> Samples() const { return make_span(samples_); }

 private:
  std::vector<SampleDesc> samples_;
};

}  // namespace warp
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_WARP_WARP_SETUP_CUH_
