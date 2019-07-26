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
#include "dali/kernels/imgproc/warp/warp_uniform_size_impl.cuh"
#include "dali/kernels/imgproc/warp/mapping_traits.h"

namespace dali {
namespace kernels {

/// @remarks Assume HWC layout
template <typename Mapping, int ndim, typename OutputType, typename InputType,
          typename BorderValue, DALIInterpType interp>
class WarpGPU : public warp::WarpSetup<ndim, OutputType, InputType> {
  using MappingParams = warp::mapping_params_t<Mapping>;
  static_assert(std::is_pod<MappingParams>::value, "Mapping must be POD.");
  static_assert(std::is_pod<BorderValue>::value, "BorderValue must be POD.");

  using Base =  warp::WarpSetup<ndim, OutputType, InputType>;
  using SampleDesc = typename Base::SampleDesc;
  using BlockDesc = typename Base::BlockDesc;
  static_assert(ndim == 2, "Not implemented for ndim != 2");

 public:
  static constexpr int tensor_dim = ndim + 1;
  KernelRequirements Setup(KernelContext &context,
                           const InListGPU<InputType, tensor_dim> &in,
                           const InTensorGPU<MappingParams, 1> &mapping,
                           span<const TensorShape<ndim>> output_sizes,
                           BorderValue border = {}) {
    assert(in.size() == static_cast<size_t>(output_sizes.size()));
    auto out_shapes = this->GetOutputShape(in.shape, output_sizes);
    return Base::Setup(out_shapes);
  }

  void Run(KernelContext &context,
           const OutListGPU<OutputType, tensor_dim> &out,
           const InListGPU<InputType, tensor_dim> &in,
           const InTensorGPU<MappingParams, 1> &mapping,
           span<const TensorShape<ndim>> output_sizes,
           BorderValue border = {}) {
    this->ValidateOutputShape(out.shape, in.shape, output_sizes);
    this->PrepareSamples(out, in);
    SampleDesc *gpu_samples;
    BlockDesc *gpu_blocks;

    if (this->IsUniformSize()) {
      gpu_samples = context.scratchpad->ToGPU(context.gpu.stream, this->Samples());
      CUDA_CALL(cudaGetLastError());

      warp::BatchWarpUniformSize<interp, Mapping, ndim, OutputType, InputType, BorderValue>
      <<<this->GridDim(), this->BlockDim(), 0, context.gpu.stream>>>(
        gpu_samples,
        this->UniformOutputSize(),
        this->UniformBlockSize(),
        mapping.data,
        border);
      CUDA_CALL(cudaGetLastError());
    } else {
      std::tie(gpu_samples, gpu_blocks) = context.scratchpad->ToContiguousGPU(
        context.gpu.stream, this->Samples(), this->Blocks());
      CUDA_CALL(cudaGetLastError());
      warp::BatchWarpVariableSize<interp, Mapping, ndim, OutputType, InputType, BorderValue>
      <<<this->GridDim(), this->BlockDim(), 0, context.gpu.stream>>>(
        gpu_samples,
        gpu_blocks,
        mapping.data,
        border);
      CUDA_CALL(cudaGetLastError());
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_WARP_GPU_H_
