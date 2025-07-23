// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/image/paste/paste.h"

#include <utility>
#include <vector>
#include <algorithm>

namespace dali {

#define PASTE_BLOCKSIZE 512

namespace {

__global__
__launch_bounds__(PASTE_BLOCKSIZE, 1)
void BatchedPaste(
    const int N,
    const int C,
    const uint8_t* const __restrict__ fill_value,
    const uint8_t* const * const __restrict__ in_batch,
    uint8_t* const* const __restrict__ out_batch,
    const int* const __restrict__ in_out_dims_paste_yx) {
  const int n = blockIdx.x;

  constexpr int blockSize = PASTE_BLOCKSIZE;
  constexpr int nThreadsPerWave = 32;  // 1 warp per row
  constexpr int nWaves = blockSize / nThreadsPerWave;
  constexpr int MAX_C = 1024;

  __shared__ uint8_t rgb[MAX_C];
  __shared__ int jump[MAX_C];
  for (int i = threadIdx.x; i < C; i += blockDim.x) {
    rgb[i] = fill_value[i % C];
    jump[i] = (i + nThreadsPerWave) % C;
  }

  const int offset = n*6;
  const int in_H = in_out_dims_paste_yx[offset];
  const int in_W = in_out_dims_paste_yx[offset + 1];
  const int out_H = in_out_dims_paste_yx[offset + 2];
  const int out_W = in_out_dims_paste_yx[offset + 3];
  const int paste_y = in_out_dims_paste_yx[offset + 4];
  const int paste_x = in_out_dims_paste_yx[offset + 5];

  const uint8_t* const input_ptr = in_batch[n];
  uint8_t * const output_ptr = out_batch[n];

  __syncthreads();

  const int myWave = threadIdx.x / nThreadsPerWave;
  const int myId = threadIdx.x % nThreadsPerWave;

  const int paste_x_stride = paste_x * C;
  const int in_stride = in_W * C;
  const int startC = myId % C;

  for (int h = myWave; h < out_H; h += nWaves) {
    const int H = h * out_W * C;
    const int in_h = h - paste_y;
    const bool h_in_range = in_h >= 0 && in_h < in_H;
    if (h_in_range) {
      int c = startC;
      for (int i = myId; i < paste_x * C; i += nThreadsPerWave) {
        const int out_idx = H + i;
        output_ptr[out_idx] = rgb[c];
        c = jump[c];
      }
      const int current_in_stride = in_h*in_stride - paste_x_stride;
      for (int i = myId + paste_x_stride; i < paste_x_stride + in_W * C; i += nThreadsPerWave) {
        const int out_idx = H + i;
        const int in_idx = current_in_stride + i;

        output_ptr[out_idx] = input_ptr[in_idx];
      }
      c = startC;
      for (int i = myId + (paste_x + in_W) * C; i < out_W * C; i += nThreadsPerWave) {
        const int out_idx = H + i;
        output_ptr[out_idx] = rgb[c];
        c = jump[c];
      }
    } else {
      int c = startC;
      for (int i = myId; i < out_W * C; i += nThreadsPerWave) {
        const int out_idx = H + i;
        output_ptr[out_idx] = rgb[c];
        c = jump[c];
      }
    }
  }
}

}  // namespace


template<>
void Paste<GPUBackend>::RunHelper(Workspace &ws) {
  auto curr_batch_size = ws.GetInputBatchSize(0);
  fill_value_.set_order(ws.stream());
  BatchedPaste<<<curr_batch_size, PASTE_BLOCKSIZE, 0, ws.stream()>>>(
      curr_batch_size,
      C_,
      fill_value_.template data<uint8_t>(),
      input_ptrs_gpu_.template data<const uint8_t*>(),
      output_ptrs_gpu_.template data<uint8_t*>(),
      in_out_dims_paste_yx_gpu_.template data<int>());
}

template<>
void Paste<GPUBackend>::SetupGPUPointers(Workspace &ws) {
  auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  auto curr_batch_size = ws.GetInputBatchSize(0);

  for (int i = 0; i < curr_batch_size; ++i) {
      input_ptrs_.template mutable_data<const uint8_t*>()[i] =
            input.template tensor<uint8_t>(i);
      output_ptrs_.template mutable_data<uint8_t*>()[i] =
            output.template mutable_tensor<uint8_t>(i);
  }

  // Copy pointers on the GPU for fast access
  input_ptrs_gpu_.Copy(input_ptrs_, ws.stream());
  output_ptrs_gpu_.Copy(output_ptrs_, ws.stream());
  in_out_dims_paste_yx_gpu_.Copy(in_out_dims_paste_yx_, ws.stream());
}

DALI_REGISTER_OPERATOR(Paste, Paste<GPUBackend>, GPU);

}  // namespace dali
