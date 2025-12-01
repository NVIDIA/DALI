// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_REMAP_REMAP_CUH_
#define DALI_OPERATORS_IMAGE_REMAP_REMAP_CUH_

#include <vector>
#include "dali/core/cuda_stream_pool.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"
#include "include/dali/core/static_switch.h"
#include "dali/kernels/dynamic_scratchpad.h"

namespace dali {
namespace remap {
namespace detail {

template<typename T>
std::enable_if_t <std::is_floating_point_v<T>>
__global__
shift_pixel_origin_per_batch(T **data, const size_t *sample_sizes, size_t n_samples,
                             T shift_value) {
  for (int sample_id = blockIdx.y;
       sample_id < static_cast<int>(n_samples);
       sample_id += gridDim.y) {
    for (int component_id = blockIdx.x * blockDim.x + threadIdx.x;
         component_id < static_cast<int>(sample_sizes[sample_id]);
         component_id += blockDim.x * gridDim.x) {
      data[sample_id][component_id] += shift_value;
    }
  }
}


template<typename T>
void
invoke_kernel_per_batch(T **data_buffers, const size_t *sample_sizes, int n_samples,
                        size_t max_sample_size, T shift_value, cudaStream_t stream) {
  static constexpr int kBlockSize = 1024;
  static constexpr float kOneOverBlockSize = 1.f / static_cast<float>(kBlockSize);
  auto max_blocks = static_cast<int>((max_sample_size + kBlockSize - 1) * kOneOverBlockSize);
  dim3 block_size(kBlockSize);
  dim3 grid_size(std::min(max_blocks, 64), std::min(n_samples, 64));
  shift_pixel_origin_per_batch<<<grid_size, block_size, 0, stream>>>
          (data_buffers, sample_sizes, n_samples, shift_value);
  CUDA_CALL(cudaGetLastError());
}


template<typename StorageBackend, typename T, int ndims>
void ShiftPixelOrigin(TensorListView <StorageBackend, T, ndims> tlv, T value,
                      dali::kernels::Scratchpad &scratch, cudaStream_t stream) {
  static_assert(std::is_floating_point_v<T>,
                "Shifting should be conducted on floating point data.");
  static_assert(is_gpu_accessible<StorageBackend>::value, "Data must be GPU-accessible.");
  auto n_samples = tlv.num_samples();
  BatchVector<size_t> sample_sizes_vec;
  sample_sizes_vec.resize(n_samples);
  for (int i = 0; i < n_samples; i++) {
    sample_sizes_vec[i] = volume(tlv.tensor_shape_span(i));
  }
  auto max_sample_size = *std::max_element(sample_sizes_vec.begin(), sample_sizes_vec.end());
  auto [gpu_tlv, gpu_sizes] = scratch.ToContiguousGPU(stream, tlv.data, sample_sizes_vec);
  invoke_kernel_per_batch(gpu_tlv, gpu_sizes, n_samples, max_sample_size, value, stream);
}

}  // namespace detail
}  // namespace remap
}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_REMAP_REMAP_CUH_
