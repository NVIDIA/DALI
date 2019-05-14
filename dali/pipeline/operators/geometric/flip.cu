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

#include <cuda_runtime_api.h>
#include "dali/pipeline/operators/geometric/flip.h"

namespace dali {

template <>
Flip<GPUBackend>::Flip(const OpSpec &spec) : Operator<GPUBackend>(spec), spec_(spec) {}

template <bool flipX, bool flipY, typename T>
__global__ void FlipKernel(T *__restrict__ output, const T *__restrict__ input, size_t layers,
                           size_t height, size_t width, size_t channels_per_layer) {
  size_t xc = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t z = blockIdx.z * blockDim.z + threadIdx.z;
  if (xc >= width * channels_per_layer || y >= height || z >= layers) {
    return;
  }
  size_t channel = xc % channels_per_layer;
  size_t x = xc / channels_per_layer;
  size_t in_x = flipX ? width - 1 - x : x;
  size_t in_y = flipY ? height - 1 - y : y;
  size_t input_idx = channel + channels_per_layer * (in_x + width * (in_y + height * z));
  size_t output_idx = channel + channels_per_layer * (x + width * (y + height * z));
  output[output_idx] = input[input_idx];
}

template <bool flipX, bool flipY>
void RunKernel(TensorList<GPUBackend> &output, const TensorList<GPUBackend> &input,
               cudaStream_t stream, size_t idx) {
  DALI_TYPE_SWITCH(
      input.type().id(), DType, const auto *input_ptr = input.tensor<DType>(idx);
      auto *output_ptr = output.mutable_tensor<DType>(idx);
      int64_t height, width, channels, layers;
      DALI_ENFORCE(input.tensor_shape(idx).size() == 3);
      if (input.GetLayout() == DALI_NHWC) {
        height = input.tensor_shape(idx)[0];
        width = input.tensor_shape(idx)[1];
        channels = input.tensor_shape(idx)[2];
        layers = 1;
      } else {
        height = input.tensor_shape(idx)[1];
        width = input.tensor_shape(idx)[2];
        channels = input.tensor_shape(idx)[0];
        layers = channels;
      }
      unsigned int block_x = width * channels / layers < 32 ? width * channels / layers : 32;
      unsigned int block_y = width < 32 ? width : 32; dim3 block(block_x, block_y, 1);
      dim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y, layers);
      FlipKernel<flipX, flipY><<<grid, block, 0, stream>>>(output_ptr, input_ptr, layers, height,
                                                           width, channels / layers);
  )
}

template <>
void Flip<GPUBackend>::RunImpl(Workspace<GPUBackend> *ws, const int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);
  auto &output = ws->Output<GPUBackend>(idx);
  DALI_ENFORCE(input.GetLayout() == DALI_NHWC || input.GetLayout() == DALI_NCHW);
  output.SetLayout(input.GetLayout());
  output.set_type(input.type());
  output.ResizeLike(input);
  auto stream = ws->stream();
  for (size_t i = 0; i < input.ntensor(); ++i) {
    auto _horizontal = GetHorizontal(ws, i);
    auto _vertical = GetVertical(ws, i);
    if (_horizontal && _vertical) {
      RunKernel<true, true>(output, input, stream, i);
    } else if (_horizontal) {
      RunKernel<true, false>(output, input, stream, i);
    } else if (_vertical) {
      RunKernel<false, true>(output, input, stream, i);
    } else {
      RunKernel<false, false>(output, input, stream, i);
    }
  }
}

DALI_REGISTER_OPERATOR(Flip, Flip<GPUBackend>, GPU);

}  // namespace dali
