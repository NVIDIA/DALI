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

#include "dali/pipeline/operators/random_paste/random_paste.h"

#include <utility>
#include <vector>
#include <algorithm>

namespace dali {

namespace {

__global__ void BatchedRandomPaste(
    const int N,
    const int C,
    const uint8 r,
    const uint8 g,
    const uint8 b,
    const uint8* const * in_batch,
    uint8* const* out_batch,
    const int* in_out_dims_paste_yx) {
  const int n = blockIdx.x;

  const int offset = n*6;
  const int in_H = in_out_dims_paste_yx[offset];
  const int in_W = in_out_dims_paste_yx[offset + 1];
  const int out_H = in_out_dims_paste_yx[offset + 2];
  const int out_W = in_out_dims_paste_yx[offset + 3];
  const int paste_y = in_out_dims_paste_yx[offset + 4];
  const int paste_x = in_out_dims_paste_yx[offset + 5];

  const uint8 rgb[] = {r, g, b};

  const uint8* input_ptr = in_batch[n];
  uint8 * const output_ptr = out_batch[n];

  for (int h = threadIdx.y; h < out_H; h += blockDim.y) {
    for (int w = threadIdx.x; w < out_W; w += blockDim.x) {
      for (int c = 0; c < C; ++c) {
        int out_idx = h*out_W*C + w*C + c;
        if (h >= paste_y
            && h < paste_y + in_H
            && w >= paste_x
            && w < paste_x + in_W) {
          // copy image
          // TODO(spanev): benchmark outside of the loop
          int in_idx = (h - paste_y)*in_W*C + (w - paste_x)*C + c;

          output_ptr[out_idx] = input_ptr[in_idx];
        } else {
          // color
          output_ptr[out_idx] = rgb[c];
        }
      }
    }
  }
}

}  // namespace


template<>
void RandomPaste<GPUBackend>::RunHelper(DeviceWorkspace *ws) {
  BatchedRandomPaste<<<batch_size_, dim3(32, 32), 0, ws->stream()>>>(
      batch_size_,
      C_,
      static_cast<uint8>(rgb_[0]),
      static_cast<uint8>(rgb_[1]),
      static_cast<uint8>(rgb_[2]),
      input_ptrs_gpu_.template data<const uint8*>(),
      output_ptrs_gpu_.template data<uint8*>(),
      in_out_dims_paste_yx_gpu_.template data<int>());
}

template<>
void RandomPaste<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace *ws) {
  // No setup shared between input sets
}

template<>
void RandomPaste<GPUBackend>::SetupSampleParams(DeviceWorkspace *ws, const int idx) {
  auto &input = ws->Input<GPUBackend>(idx);
  auto output = ws->Output<GPUBackend>(idx);

  std::vector<Dims> output_shape(batch_size_);

  for (int i = 0; i < batch_size_; ++i) {
    std::vector<Index> input_shape = input.tensor_shape(i);
    DALI_ENFORCE(input_shape.size() == 3,
        "Expects 3-dimensional image input.");

    int H = input_shape[0];
    int W = input_shape[1];
    C_ = input_shape[2];

    float ratio = std::uniform_real_distribution<>(1, max_ratio_)(rand_gen_);

    int new_H = static_cast<int>(ratio * H);
    int new_W = static_cast<int>(ratio * W);
    output_shape[i] = {new_H, new_W, C_};

    int paste_y = std::uniform_int_distribution<>(0, new_H - H)(rand_gen_);
    int paste_x = std::uniform_int_distribution<>(0, new_W - W)(rand_gen_);

    int sample_dims_paste_yx[] = {H, W, new_H, new_W, paste_y, paste_x};
    int *sample_data = in_out_dims_paste_yx_.template mutable_data<int>() + (i*6);
    std::copy(sample_dims_paste_yx, sample_dims_paste_yx + 6, sample_data);
  }

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
void RandomPaste<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  if (idx != 0)
    CUDA_CALL(cudaStreamSynchronize(ws->stream()));
  SetupSampleParams(ws, idx);
  RunHelper(ws);
}

DALI_REGISTER_OPERATOR(RandomPaste, RandomPaste<GPUBackend>, GPU);

}  // namespace dali
