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


/**
 * Access a value at given (x, y) coordinates in a strided 2D array
 * @param buffer
 * @param x In pixels
 * @param y In pixels
 * @param pitch_bytes Offset, in bytes, between consecutive rows of the array
 * @return Value at given coordinates
 */
template<typename T>
__host__ __device__ constexpr T &
pitch_xy(T *buffer, ptrdiff_t x, ptrdiff_t y, ptrdiff_t pitch_bytes) {
  return reinterpret_cast<T *>(reinterpret_cast<uintptr_t>(buffer) + pitch_bytes * y)[x];
}

}  // namespace


__global__ void
DecodeFlowComponentKernel(const int16_t *input, float *output, size_t pitch, size_t width_px,
                          size_t height) {
  size_t x = threadIdx.x + blockIdx.x * blockDim.x;
  size_t y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= width_px || y >= height) return;
  auto value_in = pitch_xy(input, x, y, pitch);
  size_t outidx = x + width_px * y;
  output[outidx] = decode_flow_component(value_in);
}


__global__ void
RgbToRgbaKernel(const uint8_t *input, uint8_t *output, size_t pitch, size_t width_px,
                size_t height) {
  constexpr size_t in_channels = 3, out_channels = 4;
  size_t x = threadIdx.x + blockIdx.x * blockDim.x;
  size_t y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= width_px || y >= height) return;
  size_t in_idx = in_channels * x + in_channels * width_px * y;
  size_t out_idx = out_channels * x + pitch * y;
  output[out_idx] = input[in_idx];
  output[out_idx + 1] = input[in_idx + 1];
  output[out_idx + 2] = input[in_idx + 2];
  output[out_idx + 3] = 255;
}


void
RgbToRgba(const uint8_t *input, uint8_t *output, size_t pitch, size_t width_px, size_t height,
          cudaStream_t stream) {
  constexpr int out_channels = 4;
  DALI_ENFORCE(pitch >= out_channels * width_px);
  dim3 block_dim(kBlockSize, kBlockSize);
  dim3 grid_dim(num_blocks(out_channels * width_px, block_dim.x),
                num_blocks(height, block_dim.y));
  RgbToRgbaKernel<<<grid_dim, block_dim, 0, stream>>>(input, output, pitch, width_px, height);
}


void DecodeFlowComponents(const int16_t *input, float *output, size_t pitch, size_t width_px,
                          size_t height, cudaStream_t stream) {
  DALI_ENFORCE(pitch >= 2 * sizeof(float) * width_px);
  dim3 block_dim(kBlockSize, kBlockSize);
  dim3 grid_dim(num_blocks(sizeof(float) * width_px, block_dim.x),
                num_blocks(height, block_dim.y));
  DecodeFlowComponentKernel<<<grid_dim, block_dim, 0, stream>>>
          (input, output, pitch, sizeof(int16_t) * width_px, height);
}

}  // namespace kernel
}  // namespace optical_flow
}  // namespace dali

