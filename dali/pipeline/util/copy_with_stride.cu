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

#include "dali/pipeline/util/copy_with_stride.h"
#include <cuda_runtime.h>
#include <dali/core/util.h>
#include <dali/core/dev_array.h>

namespace dali {

constexpr int MAX_DIMS = 15;

__global__ void CopyWithStrideKernel(uint8_t *output, const uint8_t *input, Index size,
                                     DeviceArray<Index, MAX_DIMS> out_strides,
                                     DeviceArray<Index, MAX_DIMS> in_strides,
                                     int ndim) {
  auto out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_idx >= size)
    return;
  Index in_idx = 0;
  Index elem_offset = out_idx;
  for (int dim = 0; dim < ndim; ++dim) {
    auto n = elem_offset / out_strides[dim];
    in_idx += n * in_strides[dim];
    elem_offset -= n * out_strides[dim];
  }
  output[out_idx] = input[in_idx + elem_offset];
}

template <>
void CopyWithStride<GPUBackend>(void *output, const void *input,
                                const Index *in_strides,
                                const Index *shape,
                                int ndim,
                                size_t item_size,
                                cudaStream_t stream) {
  if (!in_strides) {
    CUDA_CALL(
      cudaMemcpyAsync(output, input, volume(shape, shape + ndim) * item_size,
                      cudaMemcpyDeviceToDevice, stream));
    return;
  }
  DeviceArray<Index, MAX_DIMS> out_strides{};
  out_strides[ndim - 1] = item_size;
  for (int i = ndim - 2; i >= 0; --i) {
    out_strides[i] = out_strides[i + 1] * shape[i + 1];
  }
  DeviceArray<Index, MAX_DIMS> in_strides_arr{};
  std::copy(in_strides, in_strides + ndim, in_strides_arr.data());
  Index size = volume(shape, shape + ndim) * item_size;
  auto blocks_num = (size + 1023) / 1024;
  auto block_size = (size < 1024) ? size : 1024;
  CopyWithStrideKernel<<<blocks_num, block_size, 0, stream>>>
      (static_cast<uint8_t*>(output), static_cast<const uint8_t*>(input),
       size, out_strides, in_strides_arr, ndim);
}

}  // namespace dali
