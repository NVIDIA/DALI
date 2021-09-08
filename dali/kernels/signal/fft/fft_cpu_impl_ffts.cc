// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/kernels/signal/fft/fft_cpu_impl_ffts.h"
#include <ffts.h>
#include <cmath>
#include <complex>
#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/signal/fft/fft_cpu_impl_utils.h"
#include "dali/kernels/common/for_axis.h"
#include "dali/kernels/common/utils.h"

namespace dali {
namespace kernels {
namespace signal {
namespace fft {
namespace impl {

namespace  {

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

}  // namespace

template <typename OutputType, typename InputType, int Dims>
KernelRequirements Fft1DImplFfts<OutputType, InputType, Dims>::Setup(
    KernelContext &context,
    const InTensorCPU<InputType, Dims> &in,
    const FftArgs &args) {
  constexpr bool is_complex_out = std::is_same<OutputType, std::complex<float>>::value;
  constexpr bool is_real_out = std::is_same<OutputType, float>::value;
  DALI_ENFORCE((is_complex_out && args.spectrum_type == FFT_SPECTRUM_COMPLEX)
            || (is_real_out && args.spectrum_type != FFT_SPECTRUM_COMPLEX),
    "Output type should be complex<float> or float depending on the requested spectrum type");

  transform_axis_ = args.transform_axis >= 0 ? args.transform_axis : Dims-1;
  DALI_ENFORCE(transform_axis_ >= 0 && transform_axis_ < Dims,
    make_string("Transform axis ", transform_axis_, " is out of bounds [0, ", Dims, ")"));

  const auto n = in.shape[transform_axis_];
  auto nfft = args.nfft > 0 ? args.nfft : n;

  KernelRequirements req;
  auto out_shape = in.shape;

  ScratchpadEstimator se;
  se.add<mm::memory_kind::host, float>(size_in_buf(nfft),  32);
  se.add<mm::memory_kind::host, float>(size_out_buf(nfft), 32);
  req.scratch_sizes = se.sizes;

  out_shape[transform_axis_] = nfft / 2 + 1;
  req.output_shapes = {TensorListShape<DynamicDimensions>({out_shape})};

  if (!plan_ || nfft != nfft_) {
    if (can_use_real_impl(nfft)) {
      plan_ = {ffts_init_1d_real(nfft, FFTS_FORWARD), ffts_free};
    } else {
      plan_ = {ffts_init_1d(nfft, FFTS_FORWARD), ffts_free};
    }
    DALI_ENFORCE(plan_ != nullptr, "Could not initialize ffts plan");
    nfft_ = nfft;
  }

  return req;
}

template <typename OutputType, typename InputType, int Dims>
void Fft1DImplFfts<OutputType, InputType, Dims>::Run(
    KernelContext &context,
    const OutTensorCPU<OutputType, Dims> &out,
    const InTensorCPU<InputType, Dims> &in,
    const FftArgs &args) {
  const auto n = in.shape[transform_axis_];

  // Those should be already calculated
  assert(transform_axis_ >= 0);
  assert(nfft_ > 0);
  assert(n <= nfft_);

  // When the nfft is larger than the window lenght, we center the window
  // (padding with zeros on both side)
  int in_win_start = n < nfft_ ? (nfft_ - n) / 2 : 0;

  bool use_real_impl = can_use_real_impl(nfft_);

  auto in_buf_sz = size_in_buf(nfft_);
  // ffts requires 32-byte aligned memory
  float* in_buf = context.scratchpad->AllocateHost<float>(in_buf_sz, 32);
  memset(in_buf, 0, in_buf_sz*sizeof(float));

  auto out_buf_sz = size_out_buf(nfft_);
  // ffts requires 32-byte aligned memory
  float* out_buf = context.scratchpad->AllocateHost<float>(out_buf_sz, 32);
  memset(out_buf, 0, out_buf_sz*sizeof(float));

  auto in_shape = in.shape;
  auto in_strides = GetStrides(in_shape);
  auto out_shape = out.shape;
  auto out_strides = GetStrides(out_shape);

  ForAxis(
    out.data, in.data, out_shape.data(), out_strides.data(), in_shape.data(), in_strides.data(),
    transform_axis_, out.dim(),
    [this, &args, use_real_impl, out_buf, in_buf, in_win_start](
      OutputType *out_data, const InputType *in_data,
      int64_t out_size, int64_t out_stride, int64_t in_size, int64_t in_stride) {
        int64_t in_idx = 0;
        if (use_real_impl) {
          for (int i = 0; i < in_size; i++) {
            in_buf[in_win_start + i] = ConvertSat<float>(in_data[in_idx]);
            in_idx += in_stride;
          }
        } else {
          for (int i = 0; i < in_size; i++) {
            int64_t off = 2 * (in_win_start + i);
            in_buf[off] = ConvertSat<float>(in_data[in_idx]);
            in_buf[off + 1] = 0.0f;
            in_idx += in_stride;
          }
        }

        ffts_execute(plan_.get(), in_buf, out_buf);

        // For complex impl, out_buf_sz contains the whole spectrum,
        // for real impl, the second half of the spectrum is ommited
        // In any case, we are interested in the first half of the spectrum only
        auto *complex_fft = reinterpret_cast<std::complex<float> *>(out_buf);
        if (args.spectrum_type == FFT_SPECTRUM_COMPLEX) {
          auto *complex_out = reinterpret_cast<std::complex<float> *>(out_data);
          for (int i = 0; i < nfft_/2+1; i++) {
            complex_out[i*out_stride] = complex_fft[i];
          }
        } else {
          MagnitudeSpectrumCalculator().Calculate(
            args.spectrum_type, out_data, complex_fft, out_size, out_stride, 1);
        }
    });
}

// 1 Dim, typically input (time), producing output (frequency)
template class Fft1DImplFfts<std::complex<float>, float, 1>;  // complex fft
template class Fft1DImplFfts<float, float, 1>;  // magnitude

// 2 Dims, typically input (channels, time), producing output (channels, frequency)
template class Fft1DImplFfts<std::complex<float>, float, 2>;
template class Fft1DImplFfts<float, float, 2>;

// 3 Dims, typically input (channels, frames, time), producing output (channels, frames, frequency)
template class Fft1DImplFfts<std::complex<float>, float, 3>;
template class Fft1DImplFfts<float, float, 3>;

}  // namespace impl
}  // namespace fft
}  // namespace signal
}  // namespace kernels
}  // namespace dali
