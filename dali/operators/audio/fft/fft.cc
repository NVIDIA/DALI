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

#include "dali/operators/audio/fft/fft.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(Fft)
    .DocStr(R"code(Fast fourier transform. Returns spectrum of audio signal, or a transformation
of the spectrum (e.g. power spectrum, complex magitude, log power spectrum).)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("nfft",
      R"code(Size of the FFT. By default nfft is selected to match the lenght of the data in the
transformation axis. The number of bins created in the output is either `nfft // 2 + 1` (positive
part of the spectrum only) for real outputs (power, log power, magnitude) and `2*nfft` for complex
spectrum (real and imaginary for both positive and negative parts of the spectrum).)code",
      -1)
    .AddOptionalArg("axis",
      R"code(Index of the dimension to be transformed to the frequency domain. By default, the
last dimension is selected.)code",
      -1)
    .AddOptionalArg("spectrum_type",
      R"code(Determines the type of the spectrum in the output. Possible values are:\n
      `complex`: Output represents interleaved real and imaginary parts of the FFT
          (r0, i0, r1, i1,...) with a total size of 2*nfft real numbers.
      `magnitude`: Output represents the complex magnitude of the FFT,
          i.e sqrt(real*real + imag*imag)
          with a total size of `nfft // 2 + 1` real numbers (positive part of the spectrum only
      `power`: Output represents the power of the complex spectrum,
          i.e. real*real + imag*imag
          with a total size of `nfft // 2 + 1` real numbers
      `log_power`: Output represents the logarithm of the power spectrum,
          i.e. 10*log10(real*real + imag*imag)
          with a total size of `nfft // 2 + 1` real numbers)code",
      "complex");

template <>
void Fft<CPUBackend>::RunImpl(Workspace<CPUBackend> &ws) {
  DALI_FAIL("not yet implemented");
}

DALI_REGISTER_OPERATOR(Fft, Fft<CPUBackend>, CPU);

}  // namespace dali
