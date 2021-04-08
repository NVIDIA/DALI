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

#include "dali/operators/decoder/nvjpeg/permute_layout.h"
#include "dali/core/static_switch.h"
#include "dali/core/util.h"
#include "dali/core/format.h"
#include "dali/core/error_handling.h"


namespace dali {

template <int64_t C, typename T>
__global__ void planar_to_interleaved(T *output, const T *input, int64_t comp_size) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= comp_size) return;
  T *out = output + C * tid;
  for (int c = 0; c < C; ++c) {
    out[c] = input[c * comp_size + tid];
  }
}

template <typename T>
__global__ void planar_rgb_to_gray(T *output, const T *input, int64_t comp_size) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= comp_size) return;
  auto r = input[tid];
  auto g = input[tid + comp_size];
  auto b = input[tid + 2 * comp_size];
  output[tid] = 0.299f * r + 0.587f * g + 0.114f * b;
}

void PlanarToInterleaved(uint8_t *output, const uint8_t *input,
                         int64_t comp_size, int64_t comp_count, cudaStream_t stream) {
  if (comp_count < 2) {
    cudaMemcpyAsync(output, input, comp_size * comp_count, cudaMemcpyDeviceToDevice, stream);
    return;
  }
  int num_blocks = div_ceil(comp_size, 1024);
  int block_size = (comp_size < 1024) ? comp_size : 1024;
  VALUE_SWITCH(comp_count, c_static, (2, 3, 4), (
    planar_to_interleaved<c_static>
      <<<num_blocks, block_size, 0, stream>>>(output, input, comp_size);
  ), DALI_FAIL(make_string("Unsupported number of components: ", comp_count)););  // NOLINT
}

void PlanarRGBToGray(uint8_t *output, const uint8_t *input,
                     int64_t comp_size, cudaStream_t stream) {
  int num_blocks = div_ceil(comp_size, 1024);
  int block_size = (comp_size < 1024) ? comp_size : 1024;
  planar_rgb_to_gray<<<num_blocks, block_size, 0, stream>>>(output, input, comp_size);
}

}  // namespace dali
