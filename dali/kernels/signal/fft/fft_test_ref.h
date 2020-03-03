// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_SIGNAL_FFT_FFT_TEST_REF_H_
#define DALI_KERNELS_SIGNAL_FFT_FFT_TEST_REF_H_

#include <complex>
#include <vector>

namespace dali {
namespace kernels {
namespace signal {
namespace fft {
namespace test {

template <typename Out, typename In>
void scramble(Out *out, const In *in, int n, int stride = 1) {
  if (n == 1) {
    *out = *in;
  } else {
    scramble(out, in, n / 2, stride * 2);
    scramble(out + stride, in + n / 2, n / 2, stride * 2);
  }
}

template <typename T, typename C = std::complex<T>>
struct ReferenceFFT {
  ReferenceFFT() = default;
  explicit ReferenceFFT(int n) : N(n) {
    init();
  }

  template <typename Input>
  void operator()(C *out, const Input *in) const {
    if (N == 1) {
      *out = *in;
      return;
    }
    scramble(out, in, N);
    fft(out, N);
  }

 private:
  void fft(C *inout, int n, int twiddle_stride = 1) const {
    if (n == 2) {
        C e = inout[0] + inout[1];
        C o = inout[0] - inout[1];
        inout[0] = e;
        inout[1] = o;
    } else {
      fft(inout,       n/2, twiddle_stride * 2);
      fft(inout + n/2, n/2, twiddle_stride * 2);
      for (int i = 0; i < n / 2; i++) {
        C e = inout[i];
        C o = inout[i + n/2] * twiddle[twiddle_stride * i];
        inout[i] =       e + o;
        inout[i + n/2] = e - o;
      }
    }
  }

  void init() {
    twiddle.resize(N / 2);
    for (int i = 0; i < N / 2; i++) {
      double a = 2 * M_PI * i / N;
      twiddle[i] = C(cos(a), -sin(a));
    }
  }

  int N = 0;
  std::vector<C> twiddle;
};


}  // namespace test
}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_FFT_FFT_TEST_REF_H_
