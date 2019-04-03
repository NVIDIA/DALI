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

#include "dali/error_handling.h"
#include "dali/pipeline/operators/optical_flow/turing_of/optical_flow_turing.h"

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


/**
 * Trigger-function for kernel. This kernel utilizes 2 things:
 * 1. Convert color type to RGBA (required by optical flow)
 * 2. Reshape data to match layout required by optical flow
 *
 * @tparam ColorConversionMethod Function, that matches signature:
 *                               __global__ void (const uint8_t*, uint8_t*, size_t, size_t, size_t)
 * @param cvtm Kernel to call
 * @param input
 * @param output
 * @param pitch Stride within output memory layout. In bytes
 * @param width_px In pixels
 * @param height
 * @param out_channels How many channels output data has?
 * @param stream Stream, within which kernel is called
 */
template<typename ColorConversionMethod>
void ConvertToOFLayout(ColorConversionMethod cvtm, const uint8_t *input, uint8_t *output,
                       size_t pitch, size_t width_px, size_t height, int out_channels,
                       cudaStream_t stream) {
  DALI_ENFORCE(pitch >= out_channels * width_px);
  dim3 block_dim(kBlockSize, kBlockSize);
  dim3 grid_dim(num_blocks(out_channels * width_px, block_dim.x),
                num_blocks(height, block_dim.y));
  cvtm<<<grid_dim, block_dim, 0, stream>>>(input, output, pitch, width_px, height);
}

}  // namespace


__global__ void
RgbToRgbaKernel(const uint8_t *__restrict__ input, uint8_t *__restrict__ output, size_t pitch,
                size_t width_px, size_t height) {
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


__global__ void
BgrToRgbaKernel(const uint8_t *__restrict__ input, uint8_t *__restrict__ output, size_t pitch,
                size_t width_px, size_t height) {
  constexpr size_t in_channels = 3, out_channels = 4;
  size_t x = threadIdx.x + blockIdx.x * blockDim.x;
  size_t y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= width_px || y >= height) return;
  size_t in_idx = in_channels * x + in_channels * width_px * y;
  size_t out_idx = out_channels * x + pitch * y;
  output[out_idx] = input[in_idx + 2];
  output[out_idx + 1] = input[in_idx + 1];
  output[out_idx + 2] = input[in_idx];
  output[out_idx + 3] = 255;
}


void RgbToRgba(const uint8_t *input, uint8_t *output, size_t pitch, size_t width_px, size_t height,
               cudaStream_t stream) {
  ConvertToOFLayout(RgbToRgbaKernel, input, output, pitch, width_px, height, 4, stream);
}


void BgrToRgba(const uint8_t *input, uint8_t *output, size_t pitch, size_t width_px, size_t height,
               cudaStream_t stream) {
  ConvertToOFLayout(BgrToRgbaKernel, input, output, pitch, width_px, height, 4, stream);
}


void Gray(const uint8_t *input, uint8_t *output, size_t pitch, size_t width_px, size_t height,
          cudaStream_t stream) {
  CUDA_CALL(cudaMemcpy2DAsync(output, pitch, input, width_px * sizeof(uint8_t),
                              width_px * sizeof(uint8_t), height, cudaMemcpyDefault, stream));
}


__global__ void
DecodeFlowComponentKernel(const int16_t *__restrict__ input, float *__restrict__ output,
                          size_t pitch, size_t width_px, size_t height) {
  size_t x = threadIdx.x + blockIdx.x * blockDim.x;
  size_t y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= width_px || y >= height) return;
  auto value_in = pitch_xy(input, x, y, pitch);
  size_t outidx = x + width_px * y;
  output[outidx] = decode_flow_component(value_in);
}


__global__ void
EncodeFlowComponentKernel(const float *__restrict__ input, int16_t *__restrict__ output,
                          size_t pitch, size_t width_px, size_t height) {
  size_t x = threadIdx.x + blockIdx.x * blockDim.x;
  size_t y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= width_px || y >= height) return;
  size_t in_idx = x + width_px * y;
  size_t out_idx = x + pitch * y;
  output[out_idx] = encode_flow_component(input[in_idx]);
}


void DecodeFlowComponents(const int16_t *input, float *output, size_t pitch, size_t width_px,
                          size_t height, cudaStream_t stream) {
  DALI_ENFORCE(pitch >= 2 * sizeof(int16_t) * width_px);
  dim3 block_dim(kBlockSize, kBlockSize);
  dim3 grid_dim(num_blocks(sizeof(float) * width_px, block_dim.x),
                num_blocks(height, block_dim.y));
  DecodeFlowComponentKernel<<<grid_dim, block_dim, 0, stream>>>(input, output, pitch,
                                                                sizeof(int16_t) * width_px, height);
}


void EncodeFlowComponents(const float *input, int16_t *output, size_t pitch, size_t width_px,
                          size_t height, cudaStream_t stream) {
  dim3 block_dim(kBlockSize, kBlockSize);
  dim3 grid_dim(num_blocks(sizeof(int16_t) * width_px, block_dim.x),
                num_blocks(height, block_dim.y));
  EncodeFlowComponentKernel<<<grid_dim, block_dim, 0, stream>>>(input, output, pitch,
                                                                width_px, height);
}

}  // namespace kernel
}  // namespace optical_flow
}  // namespace dali

