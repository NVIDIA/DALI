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
#include "dali/core/static_switch.h"
#include "dali/kernels/signal/fft/fft_cpu.h"
#include "dali/pipeline/data/views.h"

#define FFT_SUPPORTED_NDIMS (2, 3)

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
bool Fft<CPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                const workspace_t<CPUBackend> &ws) {
  output_desc.resize(1);
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  kernels::KernelContext ctx;
  auto in_shape = input.shape();
  auto nsamples = in_shape.num_samples();
  auto nthreads = ws.HasThreadPool() ? ws.GetThreadPool().size() : 1;

  // Other types not supported for  now
  using InputType = float;
  using OutputType = float;
  VALUE_SWITCH(in_shape.size(), Dims, FFT_SUPPORTED_NDIMS, (
    using FftKernel = kernels::signal::fft::Fft1DCpu<OutputType, InputType, Dims>;
    kmgr_.Initialize<FftKernel>();
    kmgr_.Resize<FftKernel>(nthreads, nsamples);
    output_desc[0].type = TypeInfo::Create<OutputType>();
    output_desc[0].shape.resize(nsamples, Dims);
    for (int i = 0; i < nsamples; i++) {
      const auto in_view = view<const InputType, Dims>(input[i]);
      auto &req = kmgr_.Setup<FftKernel>(i, ctx, in_view, fft_args_);
      output_desc[0].shape.set_tensor_shape(i, req.output_shapes[0][0].shape);
    }
  ), // NOLINT
  (
    DALI_FAIL(make_string("Unsupported number of dimensions", in_shape.size()))
  )); // NOLINT

  return true;
}

template <>
void Fft<CPUBackend>::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.InputRef<CPUBackend>(0);
  auto in_shape = input.shape();
  auto& thread_pool = ws.GetThreadPool();
  // Other types not supported for now
  using InputType = float;
  using OutputType = float;
  VALUE_SWITCH(in_shape.size(), Dims, FFT_SUPPORTED_NDIMS, (
    using FftKernel = kernels::signal::fft::Fft1DCpu<OutputType, InputType, Dims>;

    for (int i = 0; i < input.shape().num_samples(); i++) {
      thread_pool.DoWorkWithID(
        [this, &input, &output, i](int thread_id) {
          kernels::KernelContext ctx;
          auto in_view = view<const InputType, Dims>(input[i]);
          auto out_view = view<OutputType, Dims>(output[i]);
          kmgr_.Run<FftKernel>(thread_id, i, ctx, out_view, in_view, fft_args_);
        });
    }
  ) ,  // NOLINT
  (
    DALI_FAIL(make_string("Not supported number of dimensions: ", in_shape.size()))
  ));  // NOLINT

  thread_pool.WaitForWork();
}

DALI_REGISTER_OPERATOR(Fft, Fft<CPUBackend>, CPU);

}  // namespace dali
