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
    const uint8* const __restrict__ fill_value,
    const uint8* const * const __restrict__ in_batch,
    uint8* const* const __restrict__ out_batch,
    const int* const __restrict__ in_out_dims_paste_yx) {
  const int n = blockIdx.x;

  constexpr int blockSize = PASTE_BLOCKSIZE;
  constexpr int nThreadsPerWave = 32;  // 1 warp per row
  constexpr int nWaves = blockSize / nThreadsPerWave;
  constexpr int MAX_C = 1024;

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
void Paste<GPUBackend>::RunHelper(DeviceWorkspace &ws) {
  auto curr_batch_size = ws.GetInputBatchSize(0);
  BatchedPaste<<<curr_batch_size, PASTE_BLOCKSIZE, 0, ws.stream()>>>(
      curr_batch_size,
      C_,
      fill_value_.template data<uint8>(),
      input_ptrs_gpu_.template data<const uint8*>(),
      output_ptrs_gpu_.template data<uint8*>(),
      in_out_dims_paste_yx_gpu_.template data<int>());
}

template<>
void Paste<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace &ws) {
  // No setup shared between input sets
}

template<>
void Paste<GPUBackend>::SetupSampleParams(DeviceWorkspace &ws) {
  auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  auto curr_batch_size = ws.GetInputBatchSize(0);

  std::vector<TensorShape<>> output_shape(curr_batch_size);

  for (int i = 0; i < curr_batch_size; ++i) {
    auto input_shape = input.tensor_shape(i);
    DALI_ENFORCE(input_shape.size() == 3,
        "Expects 3-dimensional image input.");

    int H = input_shape[0];
    int W = input_shape[1];
    C_ = input_shape[2];

    float ratio = spec_.GetArgument<float>("ratio", &ws, i);
    DALI_ENFORCE(ratio >= 1.,
      "ratio of less than 1 is not supported");

    int new_H = static_cast<int>(ratio * H);
    int new_W = static_cast<int>(ratio * W);

    int min_canvas_size_ = spec_.GetArgument<float>("min_canvas_size", &ws, i);
    DALI_ENFORCE(min_canvas_size_ >= 0.,
      "min_canvas_size_ of less than 0 is not supported");

    new_H = std::max(new_H, static_cast<int>(min_canvas_size_));
    new_W = std::max(new_W, static_cast<int>(min_canvas_size_));

    output_shape[i] = {new_H, new_W, C_};

    float paste_x_ = spec_.GetArgument<float>("paste_x", &ws, i);
    float paste_y_ = spec_.GetArgument<float>("paste_y", &ws, i);
    DALI_ENFORCE(paste_x_ >= 0,
      "paste_x of less than 0 is not supported");
    DALI_ENFORCE(paste_x_ <= 1,
      "paste_x_ of more than 1 is not supported");
    DALI_ENFORCE(paste_y_ >= 0,
      "paste_y_ of less than 0 is not supported");
    DALI_ENFORCE(paste_y_ <= 1,
      "paste_y_ of more than 1 is not supported");
    int paste_x = paste_x_ * (new_W - W);
    int paste_y = paste_y_ * (new_H - H);

    int sample_dims_paste_yx[] = {H, W, new_H, new_W, paste_y, paste_x};
    int *sample_data = in_out_dims_paste_yx_.template mutable_data<int>() + (i*NUM_INDICES);
    std::copy(sample_dims_paste_yx, sample_dims_paste_yx + NUM_INDICES, sample_data);
  }

  output.set_type(input.type());
  output.Resize(output_shape);
  output.SetLayout("HWC");

  for (int i = 0; i < curr_batch_size; ++i) {
      input_ptrs_.template mutable_data<const uint8*>()[i] =
            input.template tensor<uint8>(i);
      output_ptrs_.template mutable_data<uint8*>()[i] =
            output.template mutable_tensor<uint8>(i);
  }

  // Copy pointers on the GPU for fast access
  input_ptrs_gpu_.Copy(input_ptrs_, ws.stream());
  output_ptrs_gpu_.Copy(output_ptrs_, ws.stream());
  in_out_dims_paste_yx_gpu_.Copy(in_out_dims_paste_yx_, ws.stream());
}

template<>
void Paste<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  SetupSampleParams(ws);
  RunHelper(ws);
}

DALI_REGISTER_OPERATOR(Paste, Paste<GPUBackend>, GPU);

}  // namespace dali
