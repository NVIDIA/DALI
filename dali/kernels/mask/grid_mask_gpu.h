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

struct Args {
  int width, height, channels;
  float ca, sa, sx, sy, ratio;
};

template <typename T>
__global__ void GridMaskKernel(T *output, const T *input, const Args args) {
  int x = 32 * blockIdx.x + threadIdx.x;
  int y = 32 * blockIdx.y + threadIdx.y;
  if (x >= args.width || y >= args.height)
    return;

  int off = args.channels * (x + y * args.width);
  float fx = -args.sx + x * args.ca - y * args.sa;
  float fy = -args.sy + x * args.sa + y * args.ca;
  float m = (fx - floor(fx) >= args.ratio) || (fy - floor(fy) >= args.ratio);
  for (int i = 0; i < args.channels; i++)
    output[off + i] = input[off + i] * m;
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

  void Run(KernelContext &ctx, OutListGPU<Type> &out, const InListGPU<Type> &in,
           const std::vector<int> &tile,
           const std::vector<float> &ratio,
           const std::vector<float> &angle,
           const std::vector<float> &sx,
           const std::vector<float> &sy) {
    /*
    int tile = 50;
    float ratio = 0.5;
    float angle = 0.2;
    float sx = 20;
    float sy = 50;
    */

    for (int i = 0; i < in.num_samples(); i++) {
      const auto &shape = in.tensor_shape(i);
      int width = shape[1];
      int channels = shape[2];
      int height = shape[0];
      detail::Args args{
        width, height, channels, 
        cos(angle[i]) / tile[i], sin(angle[i]) / tile[i], 
        sx[i] / tile[i], sy[i] / tile[i], ratio[i] };
      
      dim3 block(32, 32, 1);
      dim3 grid((width + 31) / 32, (height + 31) / 32, 1);
      detail::GridMaskKernel<<<grid, block, 0, ctx.gpu.stream>>>(
        out[i].data, in[i].data, args);
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_MASK_GRID_MASK_GPU_H
