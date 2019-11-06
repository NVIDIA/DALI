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
      std::vector<std::pair<OutputType *, const InputType *>> new_slices;
      for (auto &slice : slices) {
        auto *out_ptr = slice.first;
        auto *in_ptr = slice.second;
        for (int64_t i = 0; i < in_shape[dim]; i++) {
          new_slices.push_back({out_ptr, in_ptr});
          out_ptr += out_strides[dim];
          in_ptr += in_strides[dim];
        }
      }
      std::swap(slices, new_slices);
    }
  }
}

}  // namespace

template <typename OutputType, typename InputType, int Dims>
KernelRequirements Fft1DCpu<OutputType, InputType, Dims>::Setup(
    KernelContext &context,
    const InTensorCPU<InputType, Dims> &in,
    const FftArgs &args) {
  transform_axis_ = args.transform_axis >= 0 ? args.transform_axis : Dims-1;
  DALI_ENFORCE(transform_axis_ >= 0 && transform_axis_ < Dims,
    make_string("Transform axis ", transform_axis_, " is out of bounds [0, ", Dims, ")"));

  const auto n = in.shape[transform_axis_];
  nfft_ = args.nfft > 0 ? args.nfft : n;
  DALI_ENFORCE(nfft_ >= n, "NFFT is too small");

  KernelRequirements req;
  auto out_shape = in.shape;

  ScratchpadEstimator se;
  se.add<float>(AllocType::Host, size_in_buf(nfft_),  32);
  se.add<float>(AllocType::Host, size_out_buf(nfft_), 32);
  req.scratch_sizes = se.sizes;

  out_shape[transform_axis_] = (args.spectrum_type == FFT_SPECTRUM_COMPLEX) ?
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
  const auto n = in.shape[transform_axis_];

  // Those should be already calculated
  assert(transform_axis_ >= 0);
  assert(nfft_ > 0);
  assert(n <= nfft_);

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

  TensorShape<Dims> in_strides, out_strides;
  in_strides[Dims - 1] = 1;
  out_strides[Dims - 1] = 1;
  for (int d = Dims - 1; d > 0; d--) {
    out_strides[d - 1] = out_strides[d] * out.shape[d];
    in_strides[d - 1] = in_strides[d] * in.shape[d];
  }
  // Step in the transform axis
  auto out_stride = out_strides[transform_axis_];
  auto in_stride = in_strides[transform_axis_];

  assert(nfft_ > n);

  std::vector<std::pair<OutputType*, const InputType*>> slices;
  slices.push_back(std::make_pair(out.data, in.data));
  Get1DSlices(slices, out.shape.data(), out_strides.data(),
              in.shape.data(), in_strides.data(), args.transform_axis, Dims);
  for (auto &slice: slices) {
    OutputType* out_data = slice.first;
    const InputType* in_data = slice.second;

    int64_t in_idx = 0;
    if (use_real_impl) {
      for (int i = 0; i < n; i++) {
        in_buf[i] = ConvertSat<float>(in_data[in_idx]);
        in_idx += in_stride;
      }
    } else {
      for (int i = 0; i < n; i++) {
        in_buf[2*i] = ConvertSat<float>(in_data[in_idx]);
        in_buf[2*i+1] = 0.0f;
        in_idx += in_stride;
      }
    }

    ffts_execute(plan_.get(), in_buf, out_buf);

    // For complex impl, out_buf_sz contains the whole spectrum,
    // for real impl, the second half of the spectrum is ommited and should be constructed from the
    // first half
    const bool is_full_spectrum = !use_real_impl;

    FftCalculator calc;
    switch (args.spectrum_type) {
      case FFT_SPECTRUM_COMPLEX:
        calc.Calculate<Spectrum>(
          out_data, out_buf, nfft_, out_stride, 1, true, is_full_spectrum);
        break;

      case FFT_SPECTRUM_MAGNITUDE:
        calc.Calculate<MagnitudeSpectrum>(out_data, out_buf, nfft_, out_stride, 1);
        break;

      case FFT_SPECTRUM_POWER:
        calc.Calculate<PowerSpectrum>(out_data, out_buf, nfft_, out_stride, 1);
        break;

      case FFT_SPECTRUM_LOG_POWER:
        calc.Calculate<LogPowerSpectrum>(out_data, out_buf, nfft_, out_stride, 1);
        break;

      default:
        DALI_FAIL(make_string("output type not supported: ", args.spectrum_type));
    }
  }

}

// 2 Dims, typically input (channels, time), producing output (channels, frequency)
template class Fft1DCpu<float, float, 2>;
// 3 Dims, typically input (channels, frames, time), producing output (channels, frames, frequency)
template class Fft1DCpu<float, float, 3>;

}  // namespace fft
}  // namespace audio
}  // namespace kernels
}  // namespace dali
