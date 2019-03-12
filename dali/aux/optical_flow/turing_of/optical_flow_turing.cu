// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/aux/optical_flow/turing_of/optical_flow_turing.h"
#include "dali/error_handling.h"

namespace dali {
namespace optical_flow {
namespace kernel {

namespace {

constexpr size_t kBlockSize = 32;

/**
 * Calculating number of blocks
 * @param length In bytes
 * @param block_size
 * @return
 */
inline size_t num_blocks(size_t length, size_t block_size) {
  // Calculating ceil for ints
  return (length + block_size - 1) / block_size;
}

}  // namespace


__global__ void
DecodeFlowComponentKernel(const int16_t *input, float *output, size_t pitch, size_t width,
                          size_t height) {
  size_t x = threadIdx.x + blockIdx.x * blockDim.x;
  size_t y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= width || y >= height) return;
  size_t inidx = x + pitch * y;
  size_t outidx = x + width * y;
  output[outidx] = decode_flow_component(input[inidx]);
}


__global__ void
BgrToAbgrKernel(const uint8_t *input, uint8_t *output, size_t pitch, size_t width_px,
                size_t height) {
  size_t x = threadIdx.x + blockIdx.x * blockDim.x;
  size_t y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= width_px || y >= height) return;
  size_t in_idx = 3 * x + 3 * width_px * y;
  size_t out_idx = 4 * x + pitch * y;
  output[out_idx] = 255;
  output[out_idx + 1] = input[in_idx];
  output[out_idx + 2] = input[in_idx + 1];
  output[out_idx + 3] = input[in_idx + 2];
}


void BgrToAbgr(const uint8_t *input, uint8_t *output, size_t pitch, size_t width, size_t height) {
  DALI_ENFORCE(pitch >= width * 4 / 3);
  dim3 block_dim(kBlockSize, kBlockSize);
  dim3 grid_dim(num_blocks(width, block_dim.x), num_blocks(height, block_dim.y));
  BgrToAbgrKernel<<<grid_dim, block_dim>>>(input, output, pitch, width / 3, height);
}


void DecodeFlowComponents(const int16_t *input, float *output, size_t pitch, size_t width,
                          size_t height) {
  DALI_ENFORCE(pitch >= width);
  dim3 block_dim(kBlockSize, kBlockSize);
  dim3 grid_dim(num_blocks(width, block_dim.x),num_blocks(height, block_dim.y));
  DecodeFlowComponentKernel<<<grid_dim, block_dim>>>(input, output, pitch, width, height);
}

}  // namespace kernel
}  // namespace optical_flow
}  // namespace dali

