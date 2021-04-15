// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <cassert>
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/static_switch.h"
#include "dali/core/util.h"
#include "dali/kernels/imgproc/color_manipulation/color_space_conversion_impl.cuh"
#include "dali/operators/decoder/nvjpeg/permute_layout.h"

namespace dali {

template <int C, typename T>
__global__ void planar_to_interleaved(T *output, const T *input, int64_t comp_size) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= comp_size) return;
  T *out = output + C * tid;
  for (int c = 0; c < C; ++c) {
    out[c] = input[c * comp_size + tid];
  }
}

template <typename T>
__global__ void planar_rgb_to_bgr(T *output, const T *input, int64_t comp_size) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= comp_size) return;
  auto r = input[tid];
  auto g = input[tid + comp_size];
  auto b = input[tid + 2 * comp_size];
  T *out = output + 3 * tid;
  out[0] = b;
  out[1] = g;
  out[2] = r;
}

template <typename T>
__global__ void planar_rgb_to_ycbcr(T *output, const T *input, int64_t comp_size) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= comp_size) return;
  auto r = input[tid];
  auto g = input[tid + comp_size];
  auto b = input[tid + 2 * comp_size];
  T *out = output + 3 * tid;
  out[0] = kernels::rgb_to_y<T>({r, g, b});
  out[1] = kernels::rgb_to_cb<T>({r, g, b});
  out[2] = kernels::rgb_to_cr<T>({r, g, b});
}

template <typename T>
__global__ void planar_rgb_to_gray(T *output, const T *input, int64_t comp_size) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= comp_size) return;
  auto r = input[tid];
  auto g = input[tid + comp_size];
  auto b = input[tid + 2 * comp_size];
  output[tid] = kernels::rgb_to_y<T>({r, g, b});
}

void PlanarToInterleaved(uint8_t *output, const uint8_t *input,
                         int64_t comp_size, int64_t comp_count,
                         DALIImageType out_img_type,
                         cudaStream_t stream) {
  if (comp_count < 2) {
    cudaMemcpyAsync(output, input, comp_size * comp_count, cudaMemcpyDeviceToDevice, stream);
    return;
  }
  int num_blocks = div_ceil(comp_size, 1024);
  int block_size = (comp_size < 1024) ? comp_size : 1024;
  if (out_img_type == DALI_RGB || out_img_type == DALI_ANY_DATA) {
    VALUE_SWITCH(comp_count, c_static, (2, 3, 4), (
      planar_to_interleaved<c_static>
        <<<num_blocks, block_size, 0, stream>>>(output, input, comp_size);
    ), DALI_FAIL(make_string("Unsupported number of components: ", comp_count)););  // NOLINT
  } else if (out_img_type == DALI_BGR) {
    planar_rgb_to_bgr<<<num_blocks, block_size, 0, stream>>>(output, input, comp_size);
  } else if (out_img_type == DALI_YCbCr) {
    planar_rgb_to_ycbcr<<<num_blocks, block_size, 0, stream>>>(output, input, comp_size);
  } else {
    assert(false);
  }
}

void PlanarRGBToGray(uint8_t *output, const uint8_t *input,
                     int64_t comp_size, cudaStream_t stream) {
  int num_blocks = div_ceil(comp_size, 1024);
  int block_size = (comp_size < 1024) ? comp_size : 1024;
  planar_rgb_to_gray<<<num_blocks, block_size, 0, stream>>>(output, input, comp_size);
}

}  // namespace dali
