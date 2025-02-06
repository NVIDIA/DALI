// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/kernels/signal/decibel/to_decibels_gpu.h"
#include <cmath>
#include <complex>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/signal/decibel/decibel_calculator.h"

namespace dali {
namespace kernels {
namespace signal {

template <typename T>
struct SampleDesc {
  const T *in = nullptr;
  T *out = nullptr;
  int64_t size = 0;
};

template <typename T>
__global__ void ToDecibelsKernel(const SampleDesc<T>* sample_data,
                                 ToDecibelsArgs<T> args,
                                 const T* max_values = nullptr) {
  const int64_t block_size = blockDim.y * blockDim.x;
  const int64_t grid_size = gridDim.x * block_size;
  const int sample_idx = blockIdx.y;
  const auto sample = sample_data[sample_idx];
  const int64_t offset = block_size * blockIdx.x;
  const int64_t tid = threadIdx.y * blockDim.x + threadIdx.x;
  assert(args.ref_max == (max_values != nullptr));
  T s_ref = args.ref_max ? max_values[sample_idx] : args.s_ref;
  MagnitudeToDecibel<T> dB(args.multiplier, s_ref, args.min_ratio);
  for (int64_t idx = offset + tid; idx < sample.size; idx += grid_size) {
    sample.out[idx] = dB(sample.in[idx]);
  }
}

template <typename T>
ToDecibelsGpu<T>::~ToDecibelsGpu() = default;

template <typename T>
KernelRequirements ToDecibelsGpu<T>::Setup(KernelContext &context,
                                           const InListGPU<T, DynamicDimensions> &in) {
  auto out_shape = in.shape;
  const size_t num_samples = in.size();
  KernelRequirements req;
  req.output_shapes = {out_shape};
  return req;
}

template <typename T>
void ToDecibelsGpu<T>::Run(KernelContext &context, const OutListGPU<T, DynamicDimensions> &out,
                           const InListGPU<T, DynamicDimensions> &in, const ToDecibelsArgs<T> &args,
                           InListGPU<T, 0> max_values) {
  DALI_ENFORCE(max_values.empty() || max_values.is_contiguous(),
      "Reduce all kernel expects the output to be contiguous");
  const T* max_values_data = max_values.empty() ? nullptr : max_values[0].data;

  auto num_samples = in.size();
  auto* sample_data = context.scratchpad->AllocateHost<SampleDesc<T>>(num_samples);

  for (int i = 0; i < num_samples; i++) {
    auto &sample = sample_data[i];
    sample.out = out.tensor_data(i);
    sample.in = in.tensor_data(i);
    sample.size = volume(in.tensor_shape(i));
    assert(sample.size == volume(out.tensor_shape(i)));
  }

  auto* sample_data_gpu = context.scratchpad->AllocateGPU<SampleDesc<T>>(num_samples);
  CUDA_CALL(
    cudaMemcpyAsync(sample_data_gpu, sample_data, num_samples * sizeof(SampleDesc<T>),
                    cudaMemcpyHostToDevice, context.gpu.stream));

  dim3 block(32, 32);
  auto blocks_per_sample = std::max(32, 1024 / num_samples);
  dim3 grid(blocks_per_sample, num_samples);
  ToDecibelsKernel<T><<<grid, block, 0, context.gpu.stream>>>(
      sample_data_gpu, args, max_values_data);
}

template class ToDecibelsGpu<float>;
template class ToDecibelsGpu<double>;

}  // namespace signal
}  // namespace kernels
}  // namespace dali
