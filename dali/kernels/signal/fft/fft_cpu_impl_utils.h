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

#ifndef DALI_KERNELS_SIGNAL_FFT_FFT_CPU_IMPL_UTILS_H_
#define DALI_KERNELS_SIGNAL_FFT_FFT_CPU_IMPL_UTILS_H_

#include <utility>
#include <vector>
#include "dali/core/util.h"

namespace dali {
namespace kernels {
namespace signal {
namespace fft {
namespace impl {

struct ComplexSpectrumCalculator {
  template <typename OutputType = std::complex<float>, typename InputType = std::complex<float>>
  void Calculate(OutputType *out, const InputType *in,
                 int64_t nfft, int64_t out_stride = 1, int64_t in_stride = 1,
                 bool reconstruct_second_half = false) {
    for (int i = 0; i <= nfft / 2; i++) {
      out[i*out_stride] = in[i*in_stride];
    }

    if (reconstruct_second_half) {
      for (int i = nfft / 2 + 1; i < nfft; i++) {
        // mirroring nfft/2+1+i -> nfft/2-1-i
        out[i*out_stride] = in[(nfft - i)*in_stride].conj();
      }
    }
  }
};

struct MagnitudeSpectrumCalculator {
  template <typename OutputType = float, typename InputType = std::complex<float>>
  void Calculate(FftSpectrumType spectrum_type, OutputType *out, const InputType *in,
                 int64_t length, int64_t out_stride = 1, int64_t in_stride = 1) {
    switch (spectrum_type) {
      case FFT_SPECTRUM_MAGNITUDE:
        for (int i = 0; i < length; i++) {
          out[i*out_stride] = std::abs(in[i*in_stride]);
        }
        break;
      case FFT_SPECTRUM_POWER:
        for (int i = 0; i < length; i++) {
          out[i*out_stride] = std::norm(in[i*in_stride]);
        }
        break;
      default:
        DALI_FAIL(make_string("Not a magnitude spectrum type: ", spectrum_type));
    }
  }
};

}  // namespace impl
}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_FFT_FFT_CPU_IMPL_UTILS_H_
