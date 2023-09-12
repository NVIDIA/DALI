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
#include "dali/kernels/signal/wavelet/mother_wavelet.cuh"
#include "dali/kernels/signal/wavelet/wavelet_gpu.cuh"
#include "dali/pipeline/operator/operator.h"

namespace dali {


// only resizes kernel for specific wavelet type
template <typename T, template <typename> class W>
void ResizeWaveletKernel(kernels::KernelManager &kmgr) {
  using Kernel = kernels::signal::WaveletGpu<T, W>;
  kmgr.template Resize<Kernel>(1);
}


template <typename T>
void ResizeWaveletKernelForName(const DALIWaveletName &name, kernels::KernelManager &kmgr) {
  switch (name) {
    case DALIWaveletName::DALI_HAAR:
      using kernels::signal::HaarWavelet;
      ResizeWaveletKernel<T, HaarWavelet>(kmgr);
      break;
    case DALIWaveletName::DALI_GAUS:
      using kernels::signal::GaussianWavelet;
      ResizeWaveletKernel<T, GaussianWavelet>(kmgr);
      break;
    case DALIWaveletName::DALI_MEXH:
      using kernels::signal::MexicanHatWavelet;
      ResizeWaveletKernel<T, MexicanHatWavelet>(kmgr);
      break;
    case DALIWaveletName::DALI_MORL:
      using kernels::signal::MorletWavelet;
      ResizeWaveletKernel<T, MorletWavelet>(kmgr);
      break;
    case DALIWaveletName::DALI_SHAN:
      using kernels::signal::ShannonWavelet;
      ResizeWaveletKernel<T, ShannonWavelet>(kmgr);
      break;
    case DALIWaveletName::DALI_FBSP:
      using kernels::signal::FbspWavelet;
      ResizeWaveletKernel<T, FbspWavelet>(kmgr);
      break;
    default:
      throw std::invalid_argument("Unknown wavelet name.");
  }
}


// setups kernel for specific wavelet type
template <typename T, template <typename> class W>
dali::kernels::KernelRequirements &SetupWaveletKernel(kernels::KernelManager &kmgr,
                                                      kernels::KernelContext &ctx,
                                                      TensorListView<StorageGPU, const T> &a,
                                                      TensorListView<StorageGPU, const T> &b,
                                                      const kernels::signal::WaveletSpan<T> &span,
                                                      const std::vector<T> &args) {
  using Kernel = kernels::signal::WaveletGpu<T, W>;
  return kmgr.Setup<Kernel>(0, ctx, a, b, span, args);
}

// translates wavelet name to type and runs SetupWaveletKernel() for that type
template <typename T>
dali::kernels::KernelRequirements &SetupWaveletKernelForName(
    const DALIWaveletName &name, kernels::KernelManager &kmgr, kernels::KernelContext &ctx,
    TensorListView<StorageGPU, const T> &a, TensorListView<StorageGPU, const T> &b,
    const kernels::signal::WaveletSpan<T> &span, const std::vector<T> &args) {
  switch (name) {
    case DALIWaveletName::DALI_HAAR:
      using kernels::signal::HaarWavelet;
      return SetupWaveletKernel<T, HaarWavelet>(kmgr, ctx, a, b, span, args);
      break;
    case DALIWaveletName::DALI_GAUS:
      using kernels::signal::GaussianWavelet;
      return SetupWaveletKernel<T, GaussianWavelet>(kmgr, ctx, a, b, span, args);
      break;
    case DALIWaveletName::DALI_MEXH:
      using kernels::signal::MexicanHatWavelet;
      return SetupWaveletKernel<T, MexicanHatWavelet>(kmgr, ctx, a, b, span, args);
      break;
    case DALIWaveletName::DALI_MORL:
      using kernels::signal::MorletWavelet;
      return SetupWaveletKernel<T, MorletWavelet>(kmgr, ctx, a, b, span, args);
      break;
    case DALIWaveletName::DALI_SHAN:
      using kernels::signal::ShannonWavelet;
      return SetupWaveletKernel<T, ShannonWavelet>(kmgr, ctx, a, b, span, args);
      break;
    case DALIWaveletName::DALI_FBSP:
      using kernels::signal::FbspWavelet;
      return SetupWaveletKernel<T, FbspWavelet>(kmgr, ctx, a, b, span, args);
      break;
    default:
      throw std::invalid_argument("Unknown wavelet name.");
  }
}

// runs kernel for specific wavelet type
template <typename T, template <typename> class W>
void RunWaveletKernel(kernels::KernelManager &kmgr, kernels::KernelContext &ctx,
                      TensorListView<StorageGPU, T> &out, TensorListView<StorageGPU, const T> &a,
                      TensorListView<StorageGPU, const T> &b,
                      const kernels::signal::WaveletSpan<T> &span) {
  using Kernel = kernels::signal::WaveletGpu<T, W>;
  kmgr.Run<Kernel>(0, ctx, out, a, b, span);
}

// translates wavelet name to type and runs RunWaveletKernel() for that type
template <typename T>
void RunWaveletKernelForName(const DALIWaveletName &name, kernels::KernelManager &kmgr,
                             kernels::KernelContext &ctx, TensorListView<StorageGPU, T> &out,
                             TensorListView<StorageGPU, const T> &a,
                             TensorListView<StorageGPU, const T> &b,
                             const kernels::signal::WaveletSpan<T> &span) {
  switch (name) {
    case DALIWaveletName::DALI_HAAR:
      using kernels::signal::HaarWavelet;
      RunWaveletKernel<T, HaarWavelet>(kmgr, ctx, out, a, b, span);
      break;
    case DALIWaveletName::DALI_GAUS:
      using kernels::signal::GaussianWavelet;
      RunWaveletKernel<T, GaussianWavelet>(kmgr, ctx, out, a, b, span);
      break;
    case DALIWaveletName::DALI_MEXH:
      using kernels::signal::MexicanHatWavelet;
      RunWaveletKernel<T, MexicanHatWavelet>(kmgr, ctx, out, a, b, span);
      break;
    case DALIWaveletName::DALI_MORL:
      using kernels::signal::MorletWavelet;
      RunWaveletKernel<T, MorletWavelet>(kmgr, ctx, out, a, b, span);
      break;
    case DALIWaveletName::DALI_SHAN:
      using kernels::signal::ShannonWavelet;
      RunWaveletKernel<T, ShannonWavelet>(kmgr, ctx, out, a, b, span);
      break;
    case DALIWaveletName::DALI_FBSP:
      using kernels::signal::FbspWavelet;
      RunWaveletKernel<T, FbspWavelet>(kmgr, ctx, out, a, b, span);
      break;
    default:
      throw std::invalid_argument("Unknown wavelet name.");
  }
}

}  // namespace dali

#endif  // DALI_OPERATORS_SIGNAL_WAVELET_WAVELET_RUN_H_
