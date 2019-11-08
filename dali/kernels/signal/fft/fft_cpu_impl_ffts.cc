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

#include "dali/kernels/signal/fft/fft_cpu_impl_ffts.h"
#include <ffts/ffts.h>
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

namespace dali {
namespace kernels {
namespace signal {
namespace fft {
namespace impl {

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
  nfft_ = args.nfft > 0 ? args.nfft : n;
  DALI_ENFORCE(nfft_ >= n, "NFFT is too small");

  KernelRequirements req;
  auto out_shape = in.shape;

  ScratchpadEstimator se;
  se.add<float>(AllocType::Host, size_in_buf(nfft_),  32);
  se.add<float>(AllocType::Host, size_out_buf(nfft_), 32);
  req.scratch_sizes = se.sizes;

  if (args.spectrum_type == FFT_SPECTRUM_COMPLEX) {
    out_shape[transform_axis_] = nfft_;
  } else {
    out_shape[transform_axis_] = nfft_/2 + 1;
  }
  req.output_shapes = {TensorListShape<DynamicDimensions>({out_shape})};

  if (!plan_ || plan_n_ != nfft_) {
    if (can_use_real_impl(nfft_)) {
      plan_ = {ffts_init_1d_real(nfft_, FFTS_FORWARD), ffts_free};
    } else {
      plan_ = {ffts_init_1d(nfft_, FFTS_FORWARD), ffts_free};
    }
    DALI_ENFORCE(plan_ != nullptr, "Could not initialize ffts plan");
    plan_n_ = nfft_;
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
  bool use_real_impl = can_use_real_impl(nfft_);

  auto in_buf_sz = size_in_buf(nfft_);
  // ffts requires 32-byte aligned memory
  float* in_buf = context.scratchpad->template Allocate<float>(AllocType::Host, in_buf_sz, 32);
  memset(in_buf, 0, in_buf_sz*sizeof(float));

  auto out_buf_sz = size_out_buf(nfft_);
  // ffts requires 32-byte aligned memory
  float* out_buf = context.scratchpad->template Allocate<float>(AllocType::Host, out_buf_sz, 32);
  memset(out_buf, 0, out_buf_sz*sizeof(float));

  auto in_strides = in.shape;
  in_strides[in_strides.size()-1] = 1;
  for (int d = in_strides.size()-2; d >= 0; d--) {
    in_strides[d] = in_strides[d+1] * in.shape[d+1];
  }
  auto out_strides = out.shape;
  out_strides[out_strides.size()-1] = 1;
  for (int d = out_strides.size()-2; d >= 0; d--) {
    out_strides[d] = out_strides[d+1] * out.shape[d+1];
  }

  // Step in the transform axis
  auto out_stride = out_strides[transform_axis_];
  auto in_stride = in_strides[transform_axis_];

  std::vector<std::pair<OutputType*, const InputType*>> slices;
  slices.push_back(std::make_pair(out.data, in.data));
  Get1DSlices(slices, out.shape.data(), out_strides.data(),
              in.shape.data(), in_strides.data(), args.transform_axis, Dims);
  for (auto &slice : slices) {
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

    auto* complex_fft = reinterpret_cast<std::complex<float>*>(out_buf);
    if (args.spectrum_type == FFT_SPECTRUM_COMPLEX) {
      auto* complex_out = reinterpret_cast<std::complex<float>*>(out_data);
      ComplexSpectrumCalculator().Calculate(complex_out, complex_fft, nfft_, out_stride, 1);
    } else {
      MagnitudeSpectrumCalculator().Calculate(
          args.spectrum_type, out_data, complex_fft, nfft_, out_stride, 1);
    }
  }
}

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
