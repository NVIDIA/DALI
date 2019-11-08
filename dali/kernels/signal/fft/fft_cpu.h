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

#ifndef DALI_KERNELS_SIGNAL_FFT_FFT_CPU_H_
#define DALI_KERNELS_SIGNAL_FFT_FFT_CPU_H_

#include <memory>
#include <complex>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {
namespace signal {
namespace fft {

enum FftSpectrumType {
  FFT_SPECTRUM_COMPLEX = 0,    // separate interleaved real and img parts: (r0, i0, r1, i1, ...)
  FFT_SPECTRUM_MAGNITUDE = 1,  // sqrt( real^2 + img^2 )
  FFT_SPECTRUM_POWER = 2,      // real^2 + img^2
  FFT_SPECTRUM_LOG_POWER = 3,  // 10 * log10( real^2 + img^2 )
};

struct FftArgs {
  FftSpectrumType spectrum_type = FFT_SPECTRUM_COMPLEX;
  int transform_axis = -1;
  int nfft = -1;

  inline bool operator==(const FftArgs& oth) const {
    return spectrum_type == oth.spectrum_type &&
           transform_axis == oth.transform_axis &&
           nfft == oth.nfft;
  }

  inline bool operator!=(const FftArgs& oth) const {
    return !operator==(oth);
  }
};

namespace impl {

template <typename OutputType = std::complex<float>, typename InputType = float, int Dims = 2>
class DLL_PUBLIC FftImpl {
 public:
  static_assert(std::is_same<InputType, float>::value,
    "Data types other than float are not yet supported");

  static_assert(std::is_same<OutputType, float>::value
             || std::is_same<OutputType, std::complex<float>>::value,
    "Data types other than float are not yet supported");

  DLL_PUBLIC virtual ~FftImpl() = default;

  DLL_PUBLIC virtual KernelRequirements Setup(KernelContext &context,
                                              const InTensorCPU<InputType, Dims> &in,
                                              const FftArgs &args) = 0;

  DLL_PUBLIC virtual void Run(KernelContext &context,
                              const OutTensorCPU<OutputType, Dims> &out,
                              const InTensorCPU<InputType, Dims> &in,
                              const FftArgs &args) = 0;
};

}  // namespace impl

/**
 * @brief Computes 1-D FFT related transformation from real data to either a complex spectrum
 *   or a transformation of the complex spectrum (power, magnitude, log power)
 *
 * Input data can be a 2D or 3D tensor representing an signal (e.g. [channels, time]) or
 * a sequence of frames (e.g. [channels, frames, time]). The kernel can work with other data
 * layouts by providing the transform_axis representing the dimension to be transformed to the
 * frequency domain (e.g. for a layout of [channels, time, frames] we set transform_axis=1 to
 * produce a [channels, frequency, frames] layout)
 *
 * @param args.spectrum_type defines the nature of the output
 *   FFT_SPECTRUM_COMPLEX:
 *     Output represents the complex spectrum with real and imaginary parts interleaved
 *     Output is a 2D tensor of shape Fx(NFFT*2) where NFFT represents the FFT size
 *   FFT_SPECTRUM_MAGNITUDE:
 *     Output represents the magnitude of positive half of the spectrum,
 *      as a 2D tensor of shape Fx(NFFT/2+1)
 *   FFT_SPECTRUM_POWER:
 *     Output represents the power of the spectrum, as a 2D tensor of shape Fx(NFFT/2+1)
 *   FFT_SPECTRUM_LOG_POWER:
 *     Output represents the log power spectrum, as a 2D tensor of shape Fx(NFFT/2+1)
 * (where NFFT is the size of the FFT)
 *
 * @param args.nfft Number of samples in the FFT. If not provided, nfft will be calculated as the
 *   next power of two of the size of the transform axis.
 *
 * @param args.transform_axis Axis along which the FFT transformation will be calculated
 *     (note: current implementation only supports transform_axis to be the inner-most dimension
 */
template <typename OutputType = std::complex<float>,  typename InputType = float, int Dims = 2>
class DLL_PUBLIC Fft1DCpu {
 public:
  static_assert(std::is_same<InputType, float>::value,
    "Data types other than float are not yet supported");

  static_assert(std::is_same<OutputType, float>::value
             || std::is_same<OutputType, std::complex<float>>::value,
    "Data types other than float are not yet supported");

  DLL_PUBLIC KernelRequirements Setup(KernelContext &context,
                                      const InTensorCPU<InputType, Dims> &in,
                                      const FftArgs &args);

  DLL_PUBLIC void Run(KernelContext &context,
                      const OutTensorCPU<OutputType, Dims> &out,
                      const InTensorCPU<InputType, Dims> &in,
                      const FftArgs &args);
 private:
  using Impl = impl::FftImpl<OutputType, InputType, Dims>;
  std::unique_ptr<Impl> impl_;
  FftArgs args_;
};

}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_FFT_FFT_CPU_H_
