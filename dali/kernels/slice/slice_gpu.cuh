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

#ifndef DALI_KERNELS_SLICE_SLICE_GPU_H_
#define DALI_KERNELS_SLICE_SLICE_GPU_H_

#include <cuda_runtime.h>
#include <vector>
#include <utility>
#include "dali/kernels/slice/slice_kernel_utils.h"
#include "dali/kernels/kernel.h"
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/dev_array.h"
#include "dali/core/error_handling.h"

namespace dali {
namespace kernels {

template <std::size_t Dims>
struct SliceArgsDev {
  DeviceArray<int64_t, Dims> in_strides;
  DeviceArray<int64_t, Dims> out_strides;
  DeviceArray<int64_t, Dims> anchor;
};

template <typename OutputType, typename InputType, std::size_t Dims>
__global__ void SliceKernel(OutputType *__restrict__ output,
                            const InputType *__restrict__ input,
                            SliceArgsDev<Dims> slice_args_dev,
                            unsigned int total_pixels) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_pixels) {
    return;
  }

  unsigned int out_idx = idx;
  unsigned int in_idx = 0;
  for (std::size_t d = 0; d < Dims; d++) {
    unsigned int i_d = idx / slice_args_dev.out_strides[d];
    idx = idx % slice_args_dev.out_strides[d];
    in_idx += (slice_args_dev.anchor[d] + i_d) * slice_args_dev.in_strides[d];
  }
  output[out_idx] = clamp<OutputType>(input[in_idx]);
}

template <typename OutputType, typename InputType, std::size_t Dims>
class DLL_PUBLIC SliceGPU {
 public:
  SliceGPU() = default;

  DLL_PUBLIC KernelRequirements Setup(KernelContext &context,
                                      const InListGPU<InputType, Dims> &in,
                                      const std::vector<SliceArgs<Dims>> &slice_args) {
    KernelRequirements req;
    req.output_shapes = { GetOutputShapes<Dims>(in.shape, slice_args) };
    return req;
  }

  DLL_PUBLIC void Run(KernelContext &context,
                      OutListGPU<OutputType, Dims> &out,
                      const InListGPU<InputType, Dims> &in,
                      const std::vector<SliceArgs<Dims>> &slice_args) {
    for (int i = 0; i < in.size(); i++) {
      const auto in_shape = in.tensor_shape(i);
      const auto out_shape = out.tensor_shape(i);
      const auto &anchor = slice_args[i].anchor;
      const unsigned int total_size = volume(out_shape);
      const unsigned int block = std::min(total_size, 1024u);
      const unsigned int grid = div_ceil(total_size, block);

      SliceArgsDev<Dims> slice_args_dev;
      slice_args_dev.in_strides = GetStrides<Dims>(in_shape);
      slice_args_dev.out_strides = GetStrides<Dims>(out_shape);
      slice_args_dev.anchor = anchor;

      const InputType *in_ptr = in.tensor_data(i);
      OutputType *out_ptr = out.tensor_data(i);

      SliceKernel<OutputType, InputType, Dims>
          <<<grid, block, 0, context.gpu.stream>>>(out_ptr, in_ptr, slice_args_dev, total_size);
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_GPU_H_
