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

#include "dali/operators/decoder/inflate/inflate_gpu.h"
#include "dali/pipeline/data/views.h"

namespace dali {

namespace inflate {

namespace detail {

__global__ void FillTheTails(void* const* chunks, size_t* actual_sizes, size_t* output_sizes) {
  uint8_t* chunk_data = static_cast<uint8_t*>(chunks[blockIdx.z]);
  auto output_size = output_sizes[blockIdx.z];
  auto inflated_size = actual_sizes[blockIdx.z];
  for (auto idx = inflated_size + threadIdx.x + blockIdx.x * blockDim.x; idx < output_size;
       idx += blockDim.x * gridDim.x) {
    chunk_data[idx] = 0;
  }
}

}  // namespace detail

void FillTheTails(DALIDataType output_type, int batch_size, void* const* chunks_dev,
                  size_t* actual_sizes_dev, size_t* output_sizes_dev, cudaStream_t stream) {
  dim3 grid(1, 1, batch_size);
  detail::FillTheTails<<<grid, 128, 0, stream>>>(chunks_dev, actual_sizes_dev, output_sizes_dev);
  CUDA_CALL(cudaGetLastError());
}

}  // namespace inflate

}  // namespace dali
