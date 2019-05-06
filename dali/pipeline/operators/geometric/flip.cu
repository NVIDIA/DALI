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
#include <dali/pipeline/operators/geometric/flip.h>

namespace dali {

template <>
Flip<GPUBackend>::Flip(const OpSpec &spec)
    : Operator<GPUBackend>(spec),
      _horizontal(spec.GetArgument<int32>("horizontal")),
      _vertical(spec.GetArgument<int32>("vertical")) {}

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
__global__ void FlipKernel(T *output, const T *input, size_t height, size_t width,
                           size_t channels) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < height * width) {
    size_t r = idx / width;
    size_t c = idx % width;
    size_t out_idx = IndexFlip::OutputIdx(height, width, r, c);
    memcpy(&output[out_idx * channels], &input[(r * width + c) * channels], sizeof(T) * channels);
  }
}

template <typename IndexFlip>
void RunKernel(TensorList<GPUBackend> &output, const TensorList<GPUBackend> &input,
               cudaStream_t stream) {
  DALI_TYPE_SWITCH(input.type().id(), DType,
    for (unsigned int i = 0; i < input.ntensor(); ++i) {
      const auto *input_ptr = input.tensor<DType>(i);
      auto *output_ptr = output.mutable_tensor<DType>(i);
      if (input.GetLayout() == DALI_NHWC) {
        DALI_ENFORCE(input.tensor_shape(i).size() == 3);
        ssize_t height = input.tensor_shape(i)[0], width = input.tensor_shape(i)[1];
        const auto channels = input.tensor_shape(i)[2];
        const auto total_size = height * width;
        const unsigned int block = total_size < 1024 ? total_size : 1024;
        const unsigned int grid = (total_size + block - 1) / block;
        FlipKernel<IndexFlip>
            <<<grid, block, 0, stream>>>(output_ptr, input_ptr, height, width, channels);
      } else if (input.GetLayout() == DALI_NCHW) {
        ssize_t height = input.tensor_shape(i)[1], width = input.tensor_shape(i)[2];
        const auto channels = input.tensor_shape(i)[0];
        const auto total_size = height * width;
        const unsigned int block = total_size < 1024 ? total_size : 1024;
        const unsigned int grid = (total_size + block - 1) / block;
        for (ssize_t c = 0; c < channels; ++c) {
          auto slice_origin = c * height * width;
          FlipKernel<IndexFlip><<<grid, block, 0, stream>>>(
              output_ptr + slice_origin, input_ptr + slice_origin, height, width, 1);
        }
      }
    }
  )
}

template <>
void Flip<GPUBackend>::RunImpl(Workspace<GPUBackend> *ws, const int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);
  auto &output = ws->Output<GPUBackend>(idx);
  output.SetLayout(input.GetLayout());
  output.set_type(input.type());
  output.ResizeLike(input);
  auto stream = ws->stream();
  if (_horizontal && _vertical) {
    RunKernel<HorizontalVerticalIdxFlip>(output, input, stream);
  } else if (_horizontal) {
    RunKernel<HorizontalIdxFlip>(output, input, stream);
  } else if (_vertical) {
    RunKernel<VerticalIdxFlip>(output, input, stream);
  } else {
    output.Copy(input, stream);
  }
}

DALI_REGISTER_OPERATOR(Flip, Flip<GPUBackend>, GPU);

}  // namespace dali
