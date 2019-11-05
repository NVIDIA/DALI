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

#ifndef DALI_KERNELS_AUDIO_FFT_FFT_CPU_H_
#define DALI_KERNELS_AUDIO_FFT_FFT_CPU_H_

#include "dali/core/format.h"
#include "dali/core/common.h"
#include "dali/core/util.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include <ffts/ffts.h>

namespace dali {
namespace kernels {
namespace audio {
namespace fft {

enum FftSpectrumType {
  FFT_SPECTRUM_COMPLEX = 0,    // separate interleaved real and img parts: (r0, i0, r1, i2, ...)
  FFT_SPECTRUM_MAGNITUDE = 1,  // sqrt( real^2 + img^2 )
  FFT_SPECTRUM_POWER = 2,      // real^2 + img^2
  FFT_SPECTRUM_LOG_POWER = 3,  // 10 * log10( real^2 + img^2 )
};

struct FftArgs {
  FftSpectrumType spectrum_type = FFT_SPECTRUM_COMPLEX;
  int transform_axis = -1;
  int nfft = -1;
};

/**
 * @brief Computes 1-D FFT related transformation from real data to either a complex spectrum
 *   or a transformation of the complex spectrum (power, magnitude, log power)
 *
 * It can be typically used with a set of frames, i.e a 2D tensor where the first dimension
 * represents the frame index and the second dimension represents the dimension to be transformed
 * to the frequency domain.
 *
 * Input is typically a 2D tensor of dimensions FxN representing a set of F frames of length N
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
template <typename OutputType, typename InputType = OutputType, int Dims = 2>
class DLL_PUBLIC Fft1DCpu {
 public:
  static_assert(std::is_same<InputType, OutputType>::value
             && std::is_same<InputType, float>::value,
    "Data types other than float are not yet supported");

  DLL_PUBLIC KernelRequirements Setup(KernelContext &context,
                                      const InTensorCPU<InputType, Dims> &in,
                                      const FftArgs &args);

  DLL_PUBLIC void Run(KernelContext &context,
                      OutTensorCPU<OutputType, Dims> &out,
                      const InTensorCPU<InputType, Dims> &in,
                      const FftArgs &args);
 private:
  using FftsPlanPtr = std::unique_ptr<ffts_plan_t, decltype(&ffts_free)>;
  FftsPlanPtr plan_{nullptr, ffts_free};
  int plan_n_ = -1;
  int nfft_ = -1;
  int transform_axis_ = -1;
};

}  // namespace fft
}  // namespace audio
}  // namespace kernels
}  // namespace dali

  #endif  // DALI_KERNELS_AUDIO_FFT_FFT_CPU_H_