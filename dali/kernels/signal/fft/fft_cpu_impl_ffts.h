// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_SIGNAL_FFT_FFT_CPU_IMPL_FFTS_H_
#define DALI_KERNELS_SIGNAL_FFT_FFT_CPU_IMPL_FFTS_H_

#include <ffts.h>
#include <memory>
#include <complex>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/signal/fft/fft_cpu.h"

namespace dali {
namespace kernels {
namespace signal {
namespace fft {
namespace impl {

template <typename OutputType = std::complex<float>, typename InputType = float, int Dims = 2>
class DLL_PUBLIC Fft1DImplFfts : public FftImpl<OutputType, InputType, Dims> {
 public:
  static_assert(std::is_same<InputType, float>::value,
    "Data types other than float are not yet supported");

  static_assert(std::is_same<OutputType, float>::value
             || std::is_same<OutputType, std::complex<float>>::value,
    "Data types other than float are not yet supported");

  DLL_PUBLIC KernelRequirements Setup(KernelContext &context,
                                      const InTensorCPU<InputType, Dims> &in,
                                      const FftArgs &args) override;

  DLL_PUBLIC void Run(KernelContext &context,
                      const OutTensorCPU<OutputType, Dims> &out,
                      const InTensorCPU<InputType, Dims> &in,
                      const FftArgs &args) override;
 private:
  using FftsPlanPtr = std::unique_ptr<ffts_plan_t, decltype(&ffts_free)>;
  FftsPlanPtr plan_{nullptr, ffts_free};
  int nfft_ = -1;
  int transform_axis_ = -1;
};

}  // namespace impl
}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_FFT_FFT_CPU_IMPL_FFTS_H_
