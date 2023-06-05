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

template <typename T, template <typename> class W >
__global__ void ComputeWavelet(const SampleDesc<T>* sample_data, W<T> wavelet) {
  auto& sample = sample_data[blockIdx.z];
  auto a = sample.a[blockIdx.y];
  const int64_t block_size = blockDim.x * blockDim.y;
  const int64_t t_id = block_size * blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x;
  if (t_id >= sample.size_in) return;
  const T t = sample.span.begin + (T)t_id / sample.span.sampling_rate;
  for (int i = 0; i < sample.size_b; ++i) {
    const int64_t out_id = blockIdx.y * sample.size_b * sample.size_in + i * sample.size_in + t_id;
    auto b = sample.b[i];
    sample.out[out_id] = wavelet(t, a, b);
  }
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
  //std::vector<SampleDesc<T>> sample_data = std::vector<SampleDesc<T>>(num_samples);
  auto* sample_data = ctx.scratchpad->AllocateHost<SampleDesc<T>>(num_samples);
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
    max_size_in = std::max(max_size_in, sample.size_in);
  }

  // auto sample_data_gpu = std::get<0>(ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, sample_data));
  auto* sample_data_gpu = ctx.scratchpad->AllocateGPU<SampleDesc<T>>(num_samples);
  CUDA_CALL(
    cudaMemcpyAsync(sample_data_gpu, sample_data, num_samples * sizeof(SampleDesc<T>),
                    cudaMemcpyHostToDevice, ctx.gpu.stream));

  dim3 block(32, 32);
  const int64_t block_size = block.x * block.y;
  dim3 grid((max_size_in + block_size - 1) / block_size, max_size_a, num_samples);

  ComputeWavelet<<<grid, block, 0, ctx.gpu.stream>>>(sample_data_gpu, wavelet_);
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
template class WaveletGpu<float, DaubechiesWavelet>;
template class WaveletGpu<double, DaubechiesWavelet>;
template class WaveletGpu<float, SymletWavelet>;
template class WaveletGpu<double, SymletWavelet>;
template class WaveletGpu<float, CoifletWavelet>;
template class WaveletGpu<double, CoifletWavelet>;
template class WaveletGpu<float, MeyerWavelet>;
template class WaveletGpu<double, MeyerWavelet>;
template class WaveletGpu<float, GaussianWavelet>;
template class WaveletGpu<double, GaussianWavelet>;
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
