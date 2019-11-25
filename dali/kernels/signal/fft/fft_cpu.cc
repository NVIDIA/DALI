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

#include "dali/kernels/signal/fft/fft_cpu.h"
#include "dali/kernels/signal/fft/fft_cpu_impl_ffts.h"
#include <cmath>
#include <complex>
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {
namespace signal {
namespace fft {

template <typename OutputType, typename InputType, int Dims>
Fft1DCpu<OutputType, InputType, Dims>::~Fft1DCpu() = default;

template <typename OutputType, typename InputType, int Dims>
KernelRequirements Fft1DCpu<OutputType, InputType, Dims>::Setup(
    KernelContext &context,
    const InTensorCPU<InputType, Dims> &in,
    const FftArgs &args) {
  if (!impl_ || args != args_) {
    impl_ = std::make_unique<impl::Fft1DImplFfts<OutputType, InputType, Dims>>();
    args_ = args;
  }
  return impl_->Setup(context, in, args);
}

template <typename OutputType, typename InputType, int Dims>
void Fft1DCpu<OutputType, InputType, Dims>::Run(
    KernelContext &context,
    const OutTensorCPU<OutputType, Dims> &out,
    const InTensorCPU<InputType, Dims> &in,
    const FftArgs &args) {
  DALI_ENFORCE(impl_ != nullptr, "Setup needs to be called before Run");
  DALI_ENFORCE(args == args_, "FFT args are not the same as the ones used during Setup");
  impl_->Run(context, out, in, args);
}

// 1 Dim, typically input (time), producing output (frequency)
template class Fft1DCpu<std::complex<float>, float, 1>;  // complex fft
template class Fft1DCpu<float, float, 1>;  // magnitude

// 2 Dims, typically input (channels, time), producing output (channels, frequency)
template class Fft1DCpu<std::complex<float>, float, 2>;  // complex fft
template class Fft1DCpu<float, float, 2>;  // magnitude
// 3 Dims, typically input (channels, frames, time), producing output (channels, frames, frequency)
template class Fft1DCpu<std::complex<float>, float, 3>;  // complex fft
template class Fft1DCpu<float, float, 3>;  // magnitude

}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali
