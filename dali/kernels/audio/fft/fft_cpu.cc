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

#include "dali/kernels/audio/fft/fft_cpu.h"
#include <ffts/ffts.h>
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
namespace audio {
namespace fft {

namespace {

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
  void Calculate(OutputType *out, const InputType *fft, int64_t nfft,
                 bool output_full_spectrum = false, bool input_full_spectrum = false) {
    Impl impl;

    for (int i = 0; i <= nfft / 2; i++) {
      impl.Calculate(out[i], fft[i]);
    }

    if (output_full_spectrum) {
      if (input_full_spectrum) {
        for (int i = nfft / 2 + 1; i < nfft; i++) {
          impl.Calculate(out[i], fft[i]);
        }
      } else {
        for (int i = nfft / 2 + 1; i < nfft; i++) {
          InputType tmp = fft[nfft - i];
          impl.Calculate(out[i], {tmp.real(), -tmp.imag()});
        }
      }
    }
  }
};

struct Spectrum {
  template <typename OutputType = std::complex<float>, typename InputType = std::complex<float>>
  inline void Calculate(OutputType &out, InputType in) {
    out = in;
  }
};

struct PowerSpectrum {
  template <typename OutputType = float, typename InputType = std::complex<float>>
  inline void Calculate(OutputType &out, InputType in) {
    out = in.real()*in.real() + in.imag()*in.imag();
  }
};

struct MagnitudeSpectrum {
  template <typename OutputType = float, typename InputType = std::complex<float>>
  inline void Calculate(OutputType &out, InputType in) {
    PowerSpectrum().Calculate(out, in);
    out = sqrt(out);
  }
};

struct LogPowerSpectrum {
  template <typename OutputType = float, typename InputType = std::complex<float>>
  inline void Calculate(OutputType &out, InputType in) {
   PowerSpectrum().Calculate(out, in);
    const OutputType kEps = 1e-30;
    if (out < kEps) {
      out = kEps;
    }
    out = 10 * log10(out);
  }
};

}  // namespace

template <typename OutputType, typename InputType, int Dims>
KernelRequirements Fft1DCpu<OutputType, InputType, Dims>::Setup(
    KernelContext &context,
    const InTensorCPU<InputType, Dims> &in,
    const FftArgs &args) {
  ValidateArgs(args);

  KernelRequirements req;
  auto out_shape = in.shape;

  ScratchpadEstimator se;
  auto n = in.shape[args.transform_axis];
  nfft_ = args.nfft > 0 ? args.nfft : next_pow2(n);
  DALI_ENFORCE(nfft_ >= n, "NFFT is too small");

  se.add<float>(AllocType::Host, size_in_buf(nfft_),  32);
  se.add<float>(AllocType::Host, size_out_buf(nfft_), 32);
  req.scratch_sizes = se.sizes;

  out_shape[args.transform_axis] = (args.spectrum_type == FFT_SPECTRUM_COMPLEX) ?
    2 * nfft_ : (nfft_/2 + 1);
  req.output_shapes = {TensorListShape<DynamicDimensions>({out_shape})};

  return req;
}

template <typename OutputType, typename InputType, int Dims>
void Fft1DCpu<OutputType, InputType, Dims>::Run(
    KernelContext &context,
    OutTensorCPU<OutputType, Dims> &out,
    const InTensorCPU<InputType, Dims> &in,
    const FftArgs &args) {
  ValidateArgs(args);
  auto n = in.shape[args.transform_axis];

  bool use_real_impl = can_use_real_impl(nfft_);

  if (!plan_ || plan_n_ != nfft_) {
    if (use_real_impl) {
      plan_ = {ffts_init_1d_real(nfft_, FFTS_FORWARD), ffts_free};
    } else {
      plan_ = {ffts_init_1d(nfft_, FFTS_FORWARD), ffts_free};
    }
    DALI_ENFORCE(plan_ != nullptr, "Could not initialize ffts plan");
    plan_n_ = nfft_;
  }

  auto in_buf_sz = size_in_buf(nfft_);
  // ffts requires 32-byte aligned memory
  float* in_buf = context.scratchpad->template Allocate<float>(AllocType::Host, in_buf_sz, 32);
  memset(in_buf, 0, in_buf_sz*sizeof(float));

  auto out_buf_sz = size_out_buf(nfft_);
  // ffts requires 32-byte aligned memory
  float* out_buf = context.scratchpad->template Allocate<float>(AllocType::Host, out_buf_sz, 32);
  memset(out_buf, 0, out_buf_sz*sizeof(float));

  const InputType* in_data = in.data;
  OutputType* out_data = out.data;

  assert(nfft_ > n);
  if (use_real_impl) {
    for (int i = 0; i < n; i++) {
      in_buf[i] = ConvertSat<float>(in_data[i]);
    }
  } else {
    for (int i = 0; i < n; i++) {
      in_buf[2*i] = ConvertSat<float>(in_data[i]);
      in_buf[2*i+1] = 0.0f;
    }
  }

  ffts_execute(plan_.get(), in_buf, out_buf);

  // For complex impl, out_buf_sz contains the whole spectrum,
  // for real impl, the second half of the spectrum is ommited and should be constructed from the
  // first half
  const bool is_full_spectrum = !use_real_impl;

  FftCalculator calc;
  const auto *fft_data_complex = reinterpret_cast<std::complex<float>*>(out_buf);
  auto *out_data_complex = reinterpret_cast<std::complex<OutputType>*>(out_data);
  switch (args.spectrum_type) {
    case FFT_SPECTRUM_COMPLEX:
      calc.Calculate<Spectrum>(
        out_data_complex, fft_data_complex, nfft_, true, is_full_spectrum);
      break;

    case FFT_SPECTRUM_MAGNITUDE:
      calc.Calculate<MagnitudeSpectrum>(out_data, fft_data_complex, nfft_);
      break;

    case FFT_SPECTRUM_POWER:
      calc.Calculate<PowerSpectrum>(out_data, fft_data_complex, nfft_);
      break;

    case FFT_SPECTRUM_LOG_POWER:
      calc.Calculate<LogPowerSpectrum>(out_data, fft_data_complex, nfft_);
      break;

    default:
      DALI_FAIL(make_string("output type not supported: ", args.spectrum_type));
  }
}

template <typename OutputType, typename InputType, int Dims>
void Fft1DCpu<OutputType, InputType, Dims>::ValidateArgs(const FftArgs& args) {
  DALI_ENFORCE(args.transform_axis >= 0 && args.transform_axis < Dims,
    make_string("Transform axis ", args.transform_axis, " is out of bounds [0, ", Dims, ")"));
  DALI_ENFORCE(args.transform_axis == Dims-1,
    make_string(
      "Expected ", Dims, "D data with transform axis being the inner most dimension",
      "Received transform_axis=", args.transform_axis));
}

template class Fft1DCpu<float, float, 2>;

}  // namespace fft
}  // namespace audio
}  // namespace kernels
}  // namespace dali
