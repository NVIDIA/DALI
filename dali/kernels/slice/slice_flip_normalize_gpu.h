// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_GPU_H_
#define DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_GPU_H_

#include <tuple>
#include "dali/core/common.h"
#include "dali/core/geom/vec.h"
#include "dali/core/small_vector.h"
#include "dali/core/tensor_shape.h"
#include "dali/kernels/common/block_setup.h"
#include "dali/kernels/imgproc/roi.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

namespace slice_flip_normalize {

/**
 * @brief Slice (and/or pad), normalize, and (optionally) flip
 *
 * @tparam Out output type
 * @tparam In input type
 * @tparam spatial_ndim Number of spatial dimensions: 2, 3, or 4 (max is seq of 3D)
 * @tparam channel_dim: points to the channel dimension. Can be either first or last dimension
 *
 * @details see SliceFlipNormalizeGPU::Args
 */
template <typename Out, typename In, int spatial_ndim, int channel_dim>
class DLL_PUBLIC SliceFlipNormalizeGPU {
 public:
  static_assert(channel_dim == 0 || channel_dim == spatial_ndim,
                "Only channel first or channel last are supported");
  // static_assert(spatial_ndim == 2 || spatial_ndim == 3,
  //               "Only 2D or 3D are supported");  TODO(janton): enable 3D
  static_assert(spatial_ndim == 2, "Only 2D supported");
  static constexpr int ndim = spatial_ndim + (channel_dim >= 0);
  static constexpr int d_dim = spatial_ndim < 3 ? -1 : channel_dim == 0 ? 1 : 0;
  static constexpr int h_dim = 0 + (channel_dim == 0) + (spatial_ndim == 3);
  static constexpr int w_dim = h_dim + 1;
  /**
   * @brief Kernel arguments
   */
  struct Args {
    struct SampleArgs {
      Roi<spatial_ndim> roi;              // input region-of-interest
      vec<spatial_ndim, bool> flip;       // per-dim flip flag
      SmallVector<float, 4> mean;         // mean (as many values as num. of channels)
      SmallVector<float, 4> inv_stddev;   // reciprocal of stddev (as many as num. of channels)
      SmallVector<float, 4> fill_values;  // per-channel fill value (if more values than number
                                          // channels in the input, the channel dimension is padded
                                          // on all pixels)
    };
    // per sample args
    SmallVector<SampleArgs, 32> sample_args;
    // optionally transpose the output (e.g. HWC to CHW)
    ivec<ndim> perm;
  };

 private:
  void Fill(TensorShape<ndim> &out_sh, vec<2, int32_t> size) {
    out_sh[h_dim] = size.y;
    out_sh[w_dim] = size.x;
  }

  void Fill(TensorShape<ndim> &out_sh, vec<3, int64_t> size) {
    out_sh[d_dim] = size.z;
    out_sh[h_dim] = size.y;
    out_sh[w_dim] = size.x;
  }

  template <typename Int>
  void Fill(vec<2, Int> &size, TensorShape<ndim>& in_sh) {
    size = {static_cast<Int>(in_sh[w_dim]), static_cast<Int>(in_sh[h_dim])};
  }

  template <typename Int>
  void Fill(vec<3, Int> &size, TensorShape<ndim>& in_sh) {
    size = {static_cast<Int>(in_sh[w_dim]), static_cast<Int>(in_sh[h_dim]),
            static_cast<Int>(in_sh[d_dim])};
  }

  TensorListShape<ndim> out_shape_orig_;  // not permuted
  TensorListShape<ndim> out_shape_;
  BlockSetup<spatial_ndim, channel_dim> block_setup_;
  int nchannels_ = -1;
  int out_nchannels_ = -1;
  ivec<ndim> perm_;
  ivec<ndim> inv_perm_;

 private:
  int GetNumChannels(const TensorListShape<ndim>& sh);
  int GetOutNumChannels(const TensorListShape<ndim>& sh, const Args& args);
  std::tuple<float *, float *, Out *> SetupParams(KernelContext &ctx, const Args &args);

 public:
  ~SliceFlipNormalizeGPU() = default;

  KernelRequirements Setup(KernelContext &ctx, const TensorListShape<ndim> &sh,
                           const Args &args);

  void Run(KernelContext &ctx, const OutListGPU<Out, ndim> &out,
           const InListGPU<In, ndim> &in, const Args &args);
};

}  // namespace slice_flip_normalize

}  // namespace kernels
}  // namespace dali


#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_GPU_H_
