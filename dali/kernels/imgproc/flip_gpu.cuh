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

#ifndef DALI_KERNELS_IMGPROC_FLIP_GPU_CUH_
#define DALI_KERNELS_IMGPROC_FLIP_GPU_CUH_

#include <cuda_runtime.h>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include "dali/core/static_switch.h"

namespace dali {
namespace kernels {

constexpr int sample_ndim = 5;

namespace detail {
namespace gpu {

template <size_t C, bool Single, typename T>
__global__ void FlipKernel(T *__restrict__ output, const T *__restrict__ input,
                           TensorShape<sample_ndim> shape,
                           bool flip_z, bool flip_y, bool flip_x) {
  size_t xc = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t fz = blockIdx.z * blockDim.z + threadIdx.z;
  const size_t seq_length = shape[0], depth = shape[1], height = shape[2],
                            width = shape[3], channels = shape[4];
  if (xc >= width * channels || y >= height || fz >= depth * seq_length) {
    return;
  }
  const size_t channel = C ? xc % C : xc % channels;
  const size_t x = C ? xc / C : xc / channels;
  const size_t z = (!Single) ? fz % depth : fz;
  const size_t f = (!Single) ? fz / depth: 0;
  const size_t in_x = flip_x ? width - 1 - x : x;
  const size_t in_y = flip_y ? height - 1 - y : y;
  const size_t in_z = flip_z ? depth - 1 - z : z;
  const size_t input_idx =
      channel + (C ? C : channels) * (in_x + width * (in_y + height * (in_z + depth * f)));
  const size_t output_idx =
      channel + (C ? C : channels) * (x + width * (y + height * (z + depth * f)));
  output[output_idx] = input[input_idx];
}

template <typename T>
void FlipImpl(T *__restrict__ output, const T *__restrict__ input,
              const TensorShape<sample_ndim> &shape,
              bool flip_z, bool flip_y, bool flip_x, cudaStream_t stream) {
  auto plane_width = shape[3] * shape[4];
  unsigned int block_x = plane_width < 32 ? plane_width : 32;
  unsigned int block_y = shape[2] < 32 ? shape[2] : 32;
  dim3 block(block_x, block_y, 1);
  dim3 grid((plane_width + block_x - 1) / block_x,
            (shape[2] + block_y - 1) / block_y,
            shape[0] * shape[1]);
  if (shape[0] == 1) {
    VALUE_SWITCH(shape[4], c_channels, (1, 2, 3, 4, 5, 6, 7, 8), (
        detail::gpu::FlipKernel<c_channels, true><<<grid, block, 0, stream>>>
          (output, input, shape, flip_z, flip_y, flip_x);), (
        detail::gpu::FlipKernel<0, true><<<grid, block, 0, stream>>>
          (output, input, shape, flip_z, flip_y, flip_x);));
  } else {
    VALUE_SWITCH(shape[4], c_channels, (1, 2, 3, 4, 5, 6, 7, 8), (
        detail::gpu::FlipKernel<c_channels, false><<<grid, block, 0, stream>>>
          (output, input, shape, flip_z, flip_y, flip_x);), (
        detail::gpu::FlipKernel<0, false><<<grid, block, 0, stream>>>
          (output, input, shape, flip_z, flip_y, flip_x);));
  }
}

}  // namespace gpu
}  // namespace detail

template <typename Type>
class DLL_PUBLIC FlipGPU {
 public:
  DLL_PUBLIC KernelRequirements Setup(KernelContext &context,
                                      const InListGPU<Type, sample_ndim> &in) {
    KernelRequirements req;
    req.output_shapes = {in.shape};
    return req;
  }

  DLL_PUBLIC void Run(KernelContext &context, OutListGPU<Type, sample_ndim> &out,
                      const InListGPU<Type, sample_ndim> &in,
                      const std::vector<int> &flip_z, const std::vector<int> &flip_y,
                      const std::vector<int> &flip_x) {
    auto num_samples = static_cast<size_t>(in.num_samples());
    DALI_ENFORCE(flip_x.size() == num_samples && flip_y.size() == num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
      const auto &shape = in.tensor_shape(i);
      auto seq_length = shape[0];
      auto depth = shape[1];
      auto height = shape[2];
      auto width = shape[3];
      auto channels = shape[4];
      auto in_data = in[i].data;
      auto out_data = out[i].data;
      detail::gpu::FlipImpl(out_data, in_data, shape, flip_z[i], flip_y[i], flip_x[i],
                            context.gpu.stream);
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_FLIP_GPU_CUH_
