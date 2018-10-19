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

#include "dali/pipeline/operators/paste/paste.h"

#include <utility>
#include <vector>
#include <algorithm>

namespace dali {

#define PASTE_BLOCKSIZE 512

namespace {

__global__
__launch_bounds__(PASTE_BLOCKSIZE, 1)
void BatchedPaste(
    const int C,
    const uint8* const __restrict__ fill_value,
    const uint8* const * const __restrict__ in_batch,
    uint8* const* const __restrict__ out_batch,
    const int* const __restrict__ in_out_dims_paste_yx) {
  const int n = blockIdx.x;

  constexpr int blockSize = PASTE_BLOCKSIZE;
  constexpr int nThreadsPerWave = 32;  // 1 warp per row
  constexpr int nWaves = blockSize / nThreadsPerWave;

  __shared__ uint8 rgb[MAX_C];
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

  const uint8* const input_ptr = in_batch[n];
  uint8 * const output_ptr = out_batch[n];

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
    int c = startC;
    if (h_in_range) {
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
void Paste<GPUBackend>::RunHelper(DeviceWorkspace *ws) {
  BatchedPaste<<<batch_size_, PASTE_BLOCKSIZE, 0, ws->stream()>>>(
      C_,
      fill_value_.template data<uint8>(),
      input_ptrs_gpu_.template data<const uint8*>(),
      output_ptrs_gpu_.template data<uint8*>(),
      in_out_dims_paste_yx_gpu_.template data<int>());
}

template<>
void Paste<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace *ws) {
  // No setup shared between input sets
}

template<>
void Paste<GPUBackend>::SetupSampleParams(DeviceWorkspace *ws, const int idx) {
  auto &input = ws->Input<GPUBackend>(idx);
  auto output = ws->Output<GPUBackend>(idx);

  std::vector<Dims> output_shape(batch_size_);

  for (int i = 0; i < batch_size_; ++i) {
    std::vector<int>sample_dims_paste_yx;
    output_shape[i] = Prepare(input.tensor_shape(i), spec_, ws, i, &sample_dims_paste_yx);
    int *sample_data = in_out_dims_paste_yx_.template mutable_data<int>() + (i*NUM_INDICES);
    std::copy(sample_dims_paste_yx.begin(), sample_dims_paste_yx.end(), sample_data);
  }

  output->set_type(input.type());
  output->Resize(output_shape);
  output->SetLayout(DALI_NHWC);

  for (int i = 0; i < batch_size_; ++i) {
      input_ptrs_.template mutable_data<const uint8*>()[i] =
            input.template tensor<uint8>(i);
      output_ptrs_.template mutable_data<uint8*>()[i] =
            output->template mutable_tensor<uint8>(i);
  }

  // Copy pointers on the GPU for fast access
  input_ptrs_gpu_.Copy(input_ptrs_, ws->stream());
  output_ptrs_gpu_.Copy(output_ptrs_, ws->stream());
  in_out_dims_paste_yx_gpu_.Copy(in_out_dims_paste_yx_, ws->stream());
}

template<>
void Paste<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  if (idx != 0)
    CUDA_CALL(cudaStreamSynchronize(ws->stream()));
  SetupSampleParams(ws, idx);
  RunHelper(ws);
}

DALI_REGISTER_OPERATOR(Paste, Paste<GPUBackend>, GPU);

}  // namespace dali
