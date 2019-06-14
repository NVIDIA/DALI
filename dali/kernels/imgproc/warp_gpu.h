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

#ifndef DALI_KERNELS_IMGPROC_WARP_GPU_H_
#define DALI_KERNELS_IMGPROC_WARP_GPU_H_

#include "dali/core/common.h"
#include "dali/core/geom/vec.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/imgproc/warp/warp_setup.cuh"
#include "dali/kernels/imgproc/warp/warp_variable_size_impl.cuh"

namespace dali {
namespace kernels {

/// @remarks Assume HWC layout
template <typename Mapping, int ndim, typename OutputType, typename InputType,
          typename BorderValue, DALIInterpType interp>
struct WarpGPU : warp::WarpSetup<ndim> {
  using Base =  warp::WarpSetup<ndim>;
  static_assert(ndim == 2, "Not implemented for ndim != 2");

  static constexpr int tensor_dim = ndim + 1;
  KernelRequirements Setup(KernelContext &context,
                           const InListGPU<InputType, tensor_dim> &in,
                           span<const TensorShape<ndim>> output_sizes,
                           span<const Mapping> mapping) {
    assert(in.size() == static_cast<size_t>(output_sizes.size));
    return Base::Setup(in.shape, output_sizes);
  }

  void Run(KernelContext &context,
           const OutListGPU<OutputType, tensor_dim> &out,
           const InListGPU<InputType, tensor_dim> &in,
           span<const TensorShape<ndim>> output_sizes,
           span<const Mapping> mapping) {
    ValidateOutputShape(out.shape, in.shape, output_sizes);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_WARP_GPU_H_
