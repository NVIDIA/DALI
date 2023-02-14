// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "naive_histogram.h"

namespace naive_histogram {

using namespace ::dali;


/**
 * Computes a histogram for a single-channel image.
 *
 * @param input One-channel image.
 * @param input_size Size (in pixels) of the input image.
 * @param n_bins Number of histogram bins.
 * @param histogram Output array. Shall be allocated accordingly to `n_bins`.
 */
__global__ void naive_histogram_kernel(
        const uint8_t *input, const int input_size, const int n_bins,
        int32_t *histogram) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= input_size) return;
  auto value = input[tid];
  int bin = value % n_bins;
  atomicAdd(&histogram[bin], 1);
}


template<>
void NaiveHistogram<GPUBackend>::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);  // Input is a batch of samples.
  const auto &shape = input.shape();
  auto &output = ws.Output<GPUBackend>(0);
  for (int sample_idx = 0;
       sample_idx < shape.num_samples(); sample_idx++) {  // Iterating over all samples in a batch.
    dim3 block_size(32);
    auto input_size = volume(input.tensor_shape(sample_idx));
    dim3 grid_size((input_size + block_size.x - 1) / block_size.x);
    naive_histogram_kernel<<<grid_size, block_size, 0, ws.stream()>>>(
            input[sample_idx].data<uint8_t>(),
            input_size,
            n_histogram_bins_,
            output[sample_idx].mutable_data<int32_t>()
    );
  }
}


}  // namespace naive_histogram
