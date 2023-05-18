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

#include "dali/kernels/signal/wavelet/wavelet_gpu.cuh"
#include <cmath>
#include <complex>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/signal/wavelet/mother_wavelet.cuh"

namespace dali {
namespace kernels {
namespace signal {

template <typename T>
struct SampleDesc {
  const T *a = nullptr;
  int64_t size_a = 0;
  T *out = nullptr;
  int64_t size_out = 0;
};

template <typename T>
__global__ void ComputeWavelet(const SampleDesc<T>* sample_data,
                               T begin, T sampling_rate, T b, MotherWavelet<T> wavelet) {
  const int64_t block_size = blockDim.x * blockDim.y;
  const int64_t tid = threadIdx.y * blockDim.x + threadIdx.x;
  const T t = begin + (T)tid / sampling_rate;
  const T a = sample_data->a[blockIdx.x];
  sample_data->out[tid + blockIdx.x * block_size] = wavelet.waveletFunc(t, a, b);
}

template <typename T>
WaveletGpu<T>::~WaveletGpu() = default;

template <typename T>
KernelRequirements WaveletGpu<T>::Setup(KernelContext &context,
                                        const WaveletArgs<T> &args) {
  ScratchpadEstimator se;
  se.add<mm::memory_kind::host, SampleDesc<T>>(1);
  se.add<mm::memory_kind::device, SampleDesc<T>>(1);
  KernelRequirements req;
  req.scratch_sizes = se.sizes;
  return req;
}

template <typename T>
void WaveletGpu<T>::Run(KernelContext &context,
                        const OutListGPU<T, 1> &out,
                        const InListGPU<T, 1> &a,
                        const WaveletArgs<T> &args) {
  auto* sample_data = context.scratchpad->AllocateHost<SampleDesc<T>>(1);

  sample_data[0].out = out.tensor_data(0);
  sample_data[0].a = a.tensor_data(0);
  sample_data[0].size_a = volume(a.tensor_shape(0));
  auto in_size = (args.end - args.begin) * args.sampling_rate;
  sample_data[0].size_out =  in_size * sample_data[0].size_a;

  auto* sample_data_gpu = context.scratchpad->AllocateGPU<SampleDesc<T>>(1);
  CUDA_CALL(
    cudaMemcpyAsync(sample_data_gpu, sample_data, sizeof(SampleDesc<T>),
                    cudaMemcpyHostToDevice, context.gpu.stream));

  dim3 block(sample_data[0].size_a);
  dim3 grid(in_size);
  ComputeWavelet<T><<<grid, block, 0, context.gpu.stream>>>(
      sample_data_gpu, args.begin, args.sampling_rate, args.b, MotherWavelet<T>(args.wavelet));
}

template class WaveletGpu<float>;
template class WaveletGpu<double>;

}  // namespace signal
}  // namespace kernels
}  // namespace dali
