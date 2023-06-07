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
#include "dali/core/tensor_shape.h"

namespace dali {
namespace kernels {
namespace signal {

// computes wavelet value for each sample in specified range,
// and each a and b coeff
template <typename T, template <typename> class W >
__global__ void ComputeWavelet(const SampleDesc<T>* sample_data, W<T> wavelet) {
  // id inside block
  const int64_t b_id = threadIdx.y * blockDim.x + threadIdx.x;
  // wavelet sample id
  const int64_t t_id = blockDim.x * blockDim.y * blockIdx.x + b_id;
  auto& sample = sample_data[blockIdx.z];
  if (t_id >= sample.size_in) return;
  __shared__ T shm[1025];
  auto a = sample.a[blockIdx.y];
  auto x = std::pow(2.0, a);
  if (a == 0.0) {
    shm[b_id] = sample.in[t_id];
  }
  else {
    shm[b_id] = x * sample.in[t_id];
    shm[1024] = std::pow(2.0, a / 2.0);
  }
  for (int i = 0; i < sample.size_b; ++i) {
    const int64_t out_id = blockIdx.y * sample.size_b * sample.size_in + i * sample.size_in + t_id;
    auto b = sample.b[i];
    if (b == 0.0) {
      sample.out[out_id] = wavelet(shm[b_id]);
    }
    else {
      sample.out[out_id] = wavelet(shm[b_id] - b);
    }
    if (a != 0.0) {
      sample.out[out_id] *= shm[1024];
    }
  }
}

// translate input range information to input samples
template <typename T>
__global__ void ComputeInputSamples(const SampleDesc<T>* sample_data) {
  const int64_t t_id = blockDim.x * blockDim.y * blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x;
  auto& sample = sample_data[blockIdx.y];
  if (t_id >= sample.size_in) return;
  sample.in[t_id] = sample.span.begin + (T)t_id / sample.span.sampling_rate;
}

template <typename T, template <typename> class W >
DLL_PUBLIC KernelRequirements WaveletGpu<T, W>::Setup(KernelContext &context,
                                                      const InListGPU<T> &a,
                                                      const InListGPU<T> &b,
                                                      const WaveletSpan<T> &span,
                                                      const std::vector<T> &args) {
  ENFORCE_SHAPES(a.shape, b.shape);
  auto out_shape = this->GetOutputShape(a.shape, b.shape, span);
  KernelRequirements req;
  req.output_shapes = {out_shape};
  wavelet_ = W(args);
  return req;
}

template <typename T, template <typename> class W >
DLL_PUBLIC void WaveletGpu<T, W>::Run(KernelContext &ctx,
                                      OutListGPU<T> &out,
                                      const InListGPU<T> &a,
                                      const InListGPU<T> &b,
                                      const WaveletSpan<T> &span) {
  ENFORCE_SHAPES(a.shape, b.shape);

  auto num_samples = a.num_samples();
  std::vector<SampleDesc<T>> sample_data = std::vector<SampleDesc<T>>(num_samples);
  int64_t max_size_in = 0, max_size_a = 0;

  for (int i = 0; i < num_samples; i++) {
    auto &sample = sample_data[i];
    sample.out = out.tensor_data(i);
    sample.a = a.tensor_data(i);
    sample.size_a = a.shape.tensor_size(i);
    max_size_a = std::max(max_size_a, sample.size_a);
    sample.b = b.tensor_data(i);
    sample.size_b = b.shape.tensor_size(i);
    sample.span = span;
    sample.size_in = std::ceil((sample.span.end - sample.span.begin) * sample.span.sampling_rate);
    CUDA_CALL(cudaMalloc(&(sample.in), sizeof(T) * sample.size_in));
    max_size_in = std::max(max_size_in, sample.size_in);
  }

  auto* sample_data_gpu = std::get<0>(ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, sample_data));

  dim3 block(32, 32);
  const int64_t block_size = block.x * block.y;
  dim3 grid1((max_size_in + block_size - 1) / block_size, num_samples);
  dim3 grid2((max_size_in + block_size - 1) / block_size, max_size_a, num_samples);

  ComputeInputSamples<<<grid1, block, 0, ctx.gpu.stream>>>(sample_data_gpu);
  auto shared_mem_size = (block_size + 1) * sizeof(T);
  ComputeWavelet<<<grid2, block, shared_mem_size, ctx.gpu.stream>>>(sample_data_gpu, wavelet_);
}

template <typename T, template <typename> class W >
TensorListShape<> WaveletGpu<T, W>::GetOutputShape(const TensorListShape<> &a_shape,
                                                   const TensorListShape<> &b_shape,
                                                   const WaveletSpan<T> &span) {
  int N = a_shape.num_samples();
  int in_size = std::ceil((span.end - span.begin) * span.sampling_rate);
  TensorListShape<> out_shape(N, 3);
  TensorShape<> tshape;
  for (int i = 0; i < N; i++) {
    // output tensor will be 3-dimensional of shape: 
    //  a coeffs x b coeffs x signal samples
    tshape = TensorShape<>({a_shape.tensor_shape(i).num_elements(), b_shape.tensor_shape(i).num_elements(), in_size});
    out_shape.set_tensor_shape(i, tshape);
  }
  return out_shape;
}

template class WaveletGpu<float, HaarWavelet>;
template class WaveletGpu<double, HaarWavelet>;
template class WaveletGpu<float, MeyerWavelet>;
template class WaveletGpu<double, MeyerWavelet>;
template class WaveletGpu<float, MexicanHatWavelet>;
template class WaveletGpu<double, MexicanHatWavelet>;
template class WaveletGpu<float, MorletWavelet>;
template class WaveletGpu<double, MorletWavelet>;
template class WaveletGpu<float, ShannonWavelet>;
template class WaveletGpu<double, ShannonWavelet>;
template class WaveletGpu<float, FbspWavelet>;
template class WaveletGpu<double, FbspWavelet>;

}  // namespace signal
}  // namespace kernels
}  // namespace dali
