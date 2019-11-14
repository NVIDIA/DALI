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

#include "dali/core/util.h"
#include <utility>
#include <vector>

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
        // start mirroring nfft/2+1+i -> nfft/2-1-i
        auto tmp = in[(nfft - i)*in_stride];
        out[i*out_stride] = {tmp.real(), -tmp.imag()};
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

/**
 * @brief iterator through all the 1-dimensional slices on a given axis
 */
template <typename OutputType, typename InputType, typename Functor>
void ForAxis(OutputType *out_ptr,
             const InputType *in_ptr,
             const int64_t *out_shape,
             const int64_t *out_strides,
             const int64_t *in_shape,
             const int64_t *in_strides,
             int axis,
             int ndim,
             Functor &&func,
             int current_dim = 0) {
  if (current_dim == ndim) {
    func(out_ptr, in_ptr, out_shape[axis], out_strides[axis], in_shape[axis], in_strides[axis]);
    return;
  }

  if (axis == current_dim) {
    ForAxis(out_ptr, in_ptr, out_shape, out_strides, in_shape, in_strides,
            axis, ndim, std::forward<Functor>(func), current_dim+1);
  } else {
    for (int i = 0; i < in_shape[current_dim]; i++) {
      ForAxis(out_ptr + i * out_strides[current_dim],
              in_ptr + i * in_strides[current_dim],
              out_shape, out_strides,
              in_shape, in_strides,
              axis, ndim, std::forward<Functor>(func), current_dim+1);
    }
  }
}

}  // namespace impl
}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_FFT_FFT_CPU_IMPL_UTILS_H_
