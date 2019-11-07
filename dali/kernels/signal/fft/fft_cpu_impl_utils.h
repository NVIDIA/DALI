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

namespace dali {
namespace kernels {
namespace signal {
namespace fft {
namespace impl {

inline int64_t next_pow2(int64_t n) {
  int64_t pow2 = 2;
  while (n > pow2) {
    pow2 *= 2;
  }
  return pow2;
}

inline bool is_pow2(int64_t n) {
  return (n & (n-1)) == 0;
}

inline bool can_use_real_impl(int64_t n) {
  // Real impl can be selected when doing forward transform and n is a power of 2
  return is_pow2(n);
}

inline int64_t size_in_buf(int64_t n) {
  // Real impl input needs:    N real numbers    -> N floats
  // Complex impl input needs: N complex numbers -> 2*N floats
  return can_use_real_impl(n) ? n : 2*n;
}

inline int64_t size_out_buf(int64_t n) {
  // Real impl output needs:    (N/2)+1 complex numbers -> N+2 floats
  // Complex impl output needs: N complex numbers       -> 2*N floats
  return can_use_real_impl(n) ? n+2 : 2*n;
}

struct FftCalculator {
  template <typename Impl, typename OutputType, typename InputType>
  void Calculate(OutputType *out, const InputType *in, int64_t nfft,
                 int64_t out_stride = 1, int64_t in_stride = 1,
                 bool output_full_spectrum = false, bool input_full_spectrum = false) {
    Impl impl;

    OutputType *out_ptr = out;
    const InputType *in_ptr = in;
    for (int i = 0; i <= nfft / 2; i++) {
      impl.Calculate(out_ptr, out_stride, in_ptr, in_stride);
    }

    if (output_full_spectrum) {
      if (input_full_spectrum) {
        for (int i = nfft / 2 + 1; i < nfft; i++) {
          impl.Calculate(out_ptr, out_stride, in_ptr, in_stride);
        }
      } else {
        int64_t out_stride_complex = 2 * out_stride;
        for (int i = nfft / 2 + 1; i < nfft; i++) {
          // start mirroring nfft/2+1+k -> nfft/2-1-k
          auto *mirror_out = out + (nfft-i) * out_stride_complex;
          // real
          *out_ptr = *mirror_out;
          out_ptr += out_stride;
          // imag
          *out_ptr = -*(mirror_out+out_stride);
          out_ptr += out_stride;
        }
      }
    }
  }
};

struct Spectrum {
  template <typename OutputType = float, typename InputType = float>
  inline void Calculate(OutputType*& out, int64_t out_stride, InputType*& in, int64_t in_stride) {
    *out = *in;
    out += out_stride;
    in += in_stride;
    *out = *in;
    out += out_stride;
    in += in_stride;
  }
};

struct PowerSpectrum {
  template <typename OutputType = float, typename InputType = float>
  inline void Calculate(OutputType*& out, int64_t out_stride, InputType*& in, int64_t in_stride) {
    auto real = *in;
    in += in_stride;
    auto imag = *in;
    in += in_stride;
    *out = real*real + imag*imag;
    out += out_stride;
  }
};

struct MagnitudeSpectrum {
  template <typename OutputType = float, typename InputType = float>
  inline void Calculate(OutputType*& out, int64_t out_stride, InputType*& in, int64_t in_stride) {
    auto* current_out = out;
    PowerSpectrum().Calculate(out, out_stride, in, in_stride);
    *current_out = sqrt(*current_out);
  }
};

struct LogPowerSpectrum {
  template <typename OutputType = float, typename InputType = float>
  inline void Calculate(OutputType*& out, int64_t out_stride, InputType*& in, int64_t in_stride) {
    auto* current_out = out;
    PowerSpectrum().Calculate(out, out_stride, in, in_stride);
    const OutputType kEps = 1e-30;
    if (*current_out < kEps) {
      *current_out = kEps;
    }
    *current_out = 10 * log10(*current_out);
  }
};

template <typename OutputType, typename InputType>
void Get1DSlices(std::vector<std::pair<OutputType*, const InputType*>>& slices,
                 const int64_t *out_shape,
                 const int64_t *out_strides,
                 const int64_t *in_shape,
                 const int64_t *in_strides,
                 int axis,
                 int ndim) {
  for (int dim = 0; dim < ndim; dim++) {
    if (axis != dim) {
      int sz = slices.size();
      for (int i = 0; i < sz; i++) {
        auto &slice = slices[i];
        auto *out_ptr = slice.first;
        auto *in_ptr = slice.second;
        for (int i = 1; i < in_shape[dim]; i++) {
          out_ptr += out_strides[dim];
          in_ptr += in_strides[dim];
          slices.push_back({out_ptr, in_ptr});
        }
      }
    }
  }
}

}  // namespace impl
}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_FFT_FFT_CPU_IMPL_UTILS_H_
