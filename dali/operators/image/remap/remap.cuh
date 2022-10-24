// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KEasdasdasdEMAP_H_
#define DALI_KEasdasdasdEMAP_H_

#include "dali/core/cuda_stream_pool.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"
#include "include/dali/core/static_switch.h"

namespace dali {
namespace remap {
namespace detail {

template<typename T>
std::enable_if_t<std::is_floating_point<T>::value>
__global__ shift_pixel_origin(T *data, int size, T shift_value) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < size; i += stride) {
    data[i] += shift_value;
  }
}


template<typename T>
void invoke_kernel(T *data, int size, T shift_value, cudaStream_t stream) {
  static constexpr int kBlockSize = 256;
  static constexpr float kOneOverBlockSize = 1.f / static_cast<float>(kBlockSize);
  int num_blocks = (size + kBlockSize - 1) * kOneOverBlockSize;
  shift_pixel_origin<<<num_blocks, kBlockSize, 0, stream>>>(data, size, shift_value);
}


template<typename StorageBackend, typename T, int ndims>
void ShiftPixelOrigin(TensorListView<StorageBackend, T, ndims> tlv, T value, cudaStream_t stream) {
  static_assert(std::is_floating_point<T>::value,
                "Shifting should be conducted on floating point data.");
  if (tlv.is_contiguous()) {
    invoke_kernel(tlv.data[0], tlv.num_elements(), value, stream);
  } else {
    for (int sample_id = 0; sample_id < tlv.num_samples(); sample_id++) {
      invoke_kernel(tlv.tensor_data(sample_id), volume(tlv.template tensor_shape(sample_id)), value,
                    stream);
    }
  }
}

}  // namespace detail
}  // namespace remap
}  // namespace dali

#endif
