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

struct NoneIdxFlip {
  static __device__ size_t OutputIdx(size_t height, size_t width, size_t r, size_t c) {
    return r * width + c;
  }
};

struct HorizontalIdxFlip {
  static __device__ size_t OutputIdx(size_t height, size_t width, size_t r, size_t c) {
    return r * width + (width - c - 1);
  }
};

struct VerticalIdxFlip {
  static __device__ size_t OutputIdx(size_t height, size_t width, size_t r, size_t c) {
    return (height - r - 1) * width + c;
  }
};

struct HorizontalVerticalIdxFlip {
  static __device__ size_t OutputIdx(size_t height, size_t width, size_t r, size_t c) {
    return (height - r - 1) * width + (width - c - 1);
  }
};

template <typename IndexFlip, typename T>
__global__ void FlipKernel(T *__restrict__ output, const T *__restrict__ input, size_t height,
                           size_t width, size_t channels_per_layer) {
  size_t r = blockIdx.x * blockDim.x + threadIdx.x;
  size_t c = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < height && c < width) {
    size_t channel = (blockIdx.z * blockDim.z + threadIdx.z) % channels_per_layer;
    size_t layer = (blockIdx.z * blockDim.z + threadIdx.z) / channels_per_layer;
    size_t input_coord = r * width + c;
    size_t output_coord = IndexFlip::OutputIdx(height, width, r, c);
    size_t layer_origin = layer * height * width * channels_per_layer;
    output[layer_origin + output_coord * channels_per_layer + channel] =
        input[layer_origin + input_coord * channels_per_layer + channel];
  }
}

template <typename IndexFlip>
void RunKernel(TensorList<GPUBackend> &output, const TensorList<GPUBackend> &input,
               cudaStream_t stream, size_t i) {
  DALI_TYPE_SWITCH(
      input.type().id(), DType,
      const auto *input_ptr = input.tensor<DType>(i);
      auto *output_ptr = output.mutable_tensor<DType>(i);
      int64_t height, width, channels, channels_per_layer;
      if (input.GetLayout() == DALI_NHWC) {
        height = input.tensor_shape(i)[0];
        width = input.tensor_shape(i)[1];
        channels = input.tensor_shape(i)[2];
        channels_per_layer = channels;
      } else {
        height = input.tensor_shape(i)[1];
        width = input.tensor_shape(i)[2];
        channels = input.tensor_shape(i)[0];
        channels_per_layer = 1;
      }
      unsigned int block_x = height < 32 ? height : 32;
      unsigned int block_y = width < 32 ? width : 32; dim3 block(block_x, block_y, 1);
      dim3 grid((height + block_x - 1) / block_x, (width + block_y - 1) / block_y, channels);
      FlipKernel<IndexFlip>
        <<<grid, block, 0, stream>>>(output_ptr, input_ptr, height, width, channels_per_layer);
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
      RunKernel<HorizontalVerticalIdxFlip>(output, input, stream, i);
    } else if (_horizontal) {
      RunKernel<HorizontalIdxFlip>(output, input, stream, i);
    } else if (_vertical) {
      RunKernel<VerticalIdxFlip>(output, input, stream, i);
    } else {
      RunKernel<NoneIdxFlip>(output, input, stream, i);
    }
  }
}

DALI_REGISTER_OPERATOR(Flip, Flip<GPUBackend>, GPU);

}  // namespace dali
