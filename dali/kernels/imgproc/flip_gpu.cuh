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

#ifndef DALI_KERNELS_FLIP_FLIP_GPU_H
#define DALI_KERNELS_FLIP_FLIP_GPU_H

#include <cuda_runtime.h>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include "dali/core/static_switch.h"

namespace dali {
namespace kernels {
namespace detail {
namespace gpu {

template <size_t C, typename T>
__global__ void FlipKernel(T *__restrict__ output, const T *__restrict__ input,
                           size_t layers, size_t height, size_t width, size_t channels,
                           bool flip_z, bool flip_y, bool flip_x) {
  size_t xc = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t z = blockIdx.z * blockDim.z + threadIdx.z;
  if (xc >= width * channels || y >= height || z >= layers) {
    return;
  }
  size_t channel = C ? xc % C : xc % channels;
  size_t x = C ? xc / C : xc / channels;
  size_t in_x = flip_x ? width - 1 - x : x;
  size_t in_y = flip_y ? height - 1 - y : y;
  size_t in_z = flip_z ? layers - 1 - z : z;
  size_t input_idx = channel + (C ? C : channels) * (in_x + width * (in_y + height * in_z));
  size_t output_idx = channel + (C ? C : channels) * (x + width * (y + height * z));
  output[output_idx] = input[input_idx];
}

template <typename T>
void FlipImpl(T *__restrict__ output, const T *__restrict__ input,
              size_t layers, size_t height, size_t width, size_t channels,
              bool flip_z, bool flip_y, bool flip_x, cudaStream_t stream) {
  auto layer_width = width * channels;
  unsigned int block_x = layer_width < 32 ? layer_width : 32;
  unsigned int block_y = height < 32 ? height : 32;
  dim3 block(block_x, block_y, 1);
  dim3 grid((layer_width + block_x - 1) / block_x, (height + block_y - 1) / block_y, layers);
  VALUE_SWITCH(channels, c_channels, (1, 2, 3, 4, 5, 6, 7, 8), (
      detail::gpu::FlipKernel<c_channels><<<grid, block, 0, stream>>>
        (output, input, layers, height, width, channels, flip_z, flip_y, flip_x);), (
      detail::gpu::FlipKernel<0><<<grid, block, 0, stream>>>
        (output, input, layers, height, width, channels, flip_z, flip_y, flip_x);));
}

}  // namespace gpu
}  // namespace detail

template <typename Type>
class DLL_PUBLIC FlipGPU {
 public:
  DLL_PUBLIC KernelRequirements Setup(KernelContext &context, const InListGPU<Type, 4> &in) {
    KernelRequirements req;
    req.output_shapes = {in.shape};
    return req;
  }

  DLL_PUBLIC void Run(KernelContext &context, OutListGPU<Type, 4> &out,
                      const InListGPU<Type, 4> &in,
                      const std::vector<int> &flip_z, const std::vector<int> &flip_y,
                      const std::vector<int> &flip_x) {
    auto num_samples = static_cast<size_t>(in.num_samples());
    DALI_ENFORCE(flip_x.size() == num_samples && flip_y.size() == num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
      auto layers = in.tensor_shape(i)[0];
      auto height = in.tensor_shape(i)[1];
      auto width = in.tensor_shape(i)[2];
      auto channels = in.tensor_shape(i)[3];
      auto in_data = in[i].data;
      auto out_data = out[i].data;
      detail::gpu::FlipImpl(out_data, in_data, layers, height, width, channels,
          flip_z[i], flip_y[i], flip_x[i], context.gpu.stream);
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_FLIP_FLIP_GPU_H
