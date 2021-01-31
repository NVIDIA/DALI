// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_MASK_GRID_MASK_GPU_H
#define DALI_KERNELS_MASK_GRID_MASK_GPU_H

#include <cuda_runtime.h>
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

namespace detail {

template <typename T>
__global__ void GridMaskKernel(T *output, const T *input,
                               int width, int height, int channels) {
  int x = 32 * blockIdx.x + threadIdx.x;
  int y = 32 * blockIdx.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  int off = channels * (x + y * width);
  for (int i = 0; i < channels; i++)
    output[off + i] = input[off + i];
}

}  // namespace detail

template <typename Type>
class GridMaskGpu {
 public:
  KernelRequirements Setup(KernelContext &context, const InListGPU<Type> &in) {
    KernelRequirements req;
    req.output_shapes = {in.shape};
    return req;
  }

  void Run(KernelContext &ctx, OutListGPU<Type> &out, const InListGPU<Type> &in) {
    for (int i = 0; i < in.num_samples(); i++) {
      const auto &shape = in.tensor_shape(i);
      unsigned width = shape[1];
      unsigned channels = shape[2];
      unsigned height = shape[0];
      dim3 block(32, 32, 1);
      dim3 grid((width + 31) / 32, (height + 31) / 32, 1);
      detail::GridMaskKernel<<<grid, block, 0, ctx.gpu.stream>>>(
        out[i].data, in[i].data, width, height, channels);
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_MASK_GRID_MASK_GPU_H
