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

#ifndef DALI_OPERATORS_SIGNAL_WAVELET_WAVELET_RUN_H_
#define DALI_OPERATORS_SIGNAL_WAVELET_WAVELET_RUN_H_

#include <vector>
#include "dali/core/format.h"
#include "dali/core/geom/mat.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/kernels/signal/wavelet/mother_wavelet.cuh"
#include "dali/kernels/signal/wavelet/wavelet_gpu.cuh"

namespace dali {

// setups and runs kernel for specific wavelet type
template <typename T, template <typename> class W >
void RunWaveletKernel(kernels::KernelManager &kmgr,
                      size_t size,
                      size_t device,
                      kernels::KernelContext &ctx,
                      TensorListView<StorageGPU, T> &out,
                      TensorListView<StorageGPU, const T>  &a,
                      TensorListView<StorageGPU, const T> &b,
                      const kernels::signal::WaveletSpan<T> &span,
                      const std::vector<T> &args) {
  using Kernel = kernels::signal::WaveletGpu<T, W>;
  kmgr.template Resize<Kernel>(1);
  kmgr.Setup<Kernel>(0, ctx, a, b, span, args);
  kmgr.Run<Kernel>(0, ctx, out, a, b, span);
}

// translates wavelet name to type and runs RunWaveletKernel() for that type
template <typename T>
void RunForName(const std::string &name,
                kernels::KernelManager &kmgr,
                size_t size,
                size_t device,
                kernels::KernelContext &ctx,
                TensorListView<StorageGPU, T> &out,
                TensorListView<StorageGPU, const T> &a,
                TensorListView<StorageGPU, const T> &b,
                const kernels::signal::WaveletSpan<T> &span,
                const std::vector<T> &args) {
  if (name == "HAAR") {
    RunWaveletKernel<T, kernels::signal::HaarWavelet>(kmgr, size, device, ctx, out, a, b, span, args);
  }
  else if (name == "DB") {
    throw new std::logic_error("Not implemented.");
    RunWaveletKernel<T, kernels::signal::DaubechiesWavelet>(kmgr, size, device, ctx, out, a, b, span, args);
  }
  else if (name == "SYM") {
    throw new std::logic_error("Not implemented.");
    RunWaveletKernel<T, kernels::signal::SymletWavelet>(kmgr, size, device, ctx, out, a, b, span, args);
  }
  else if (name == "COIF") {
    throw new std::logic_error("Not implemented.");
    RunWaveletKernel<T, kernels::signal::CoifletWavelet>(kmgr, size, device, ctx, out, a, b, span, args);
  }
  else if (name == "MEY") {
    RunWaveletKernel<T, kernels::signal::MeyerWavelet>(kmgr, size, device, ctx, out, a, b, span, args);
  }
  else if (name == "GAUS") {
    throw new std::logic_error("Not implemented.");
    RunWaveletKernel<T, kernels::signal::GaussianWavelet>(kmgr, size, device, ctx, out, a, b, span, args);
  }
  else if (name == "MEXH") {
    RunWaveletKernel<T, kernels::signal::MexicanHatWavelet>(kmgr, size, device, ctx, out, a, b, span, args);
  }
  else if (name == "MORL") {
    RunWaveletKernel<T, kernels::signal::MorletWavelet>(kmgr, size, device, ctx, out, a, b, span, args);
  }
  else if (name == "SHAN") {
    RunWaveletKernel<T, kernels::signal::ShannonWavelet>(kmgr, size, device, ctx, out, a, b, span, args);
  }
  else if (name == "FBSP") {
    RunWaveletKernel<T, kernels::signal::FbspWavelet>(kmgr, size, device, ctx, out, a, b, span, args);
  }
  else {
    throw new std::invalid_argument("Unknown wavelet name.");
  }
}

} // namespace dali

#endif // DALI_OPERATORS_SIGNAL_WAVELET_RUN_H_