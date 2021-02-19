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

#include "dali/operators/signal/fft/power_spectrum.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/signal/fft/fft_cpu.h"
#include "dali/pipeline/data/views.h"

#define FFT_SUPPORTED_NDIMS (1, 2, 3)

static constexpr int kNumInputs = 1;
static constexpr int kNumOutputs = 1;

namespace dali {

DALI_SCHEMA(PowerSpectrum)
    .DocStr(R"code(Calculates power spectrum of the signal.)code")
    .NumInput(kNumInputs)
    .NumOutput(kNumOutputs)
    .AddOptionalArg<int>("nfft",
      R"code(Size of the FFT.

By default, the ``nfft`` is selected to match the length of the data in the transformation axis.

The number of bins that are created in the output is calculated with the following formula::

   nfft // 2 + 1

.. note::
  The output only represents the positive part of the spectrum.)code",
      nullptr)
    .AddOptionalArg("axis",
      R"code(Index of the dimension to be transformed to the frequency domain.

By default, the last dimension is selected.)code",
      -1)
    .AddOptionalArg("power",
      R"code(Exponent of the FFT magnitude.

The supported values are:

* ``2`` for power spectrum ``(real*real + imag*imag)``
* ``1`` for the complex magnitude ``(sqrt(real*real + imag*imag))``.)code",
      2);

template <>
bool PowerSpectrum<CPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                          const workspace_t<CPUBackend> &ws) {
  output_desc.resize(kNumOutputs);
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  kernels::KernelContext ctx;
  auto in_shape = input.shape();
  int nsamples = input.size();
  auto nthreads = ws.GetThreadPool().NumThreads();

  // Other types not supported for now
  using InputType = float;
  using OutputType = float;
  VALUE_SWITCH(in_shape.sample_dim(), Dims, FFT_SUPPORTED_NDIMS, (
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
  ), DALI_FAIL(make_string("Unsupported number of dimensions ", in_shape.size())));  // NOLINT

  return true;
}

template <>
void PowerSpectrum<CPUBackend>::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto in_shape = input.shape();
  int nsamples = input.size();
  auto& thread_pool = ws.GetThreadPool();
  // Other types not supported for now
  using InputType = float;
  using OutputType = float;
  VALUE_SWITCH(in_shape.sample_dim(), Dims, FFT_SUPPORTED_NDIMS, (
    using FftKernel = kernels::signal::fft::Fft1DCpu<OutputType, InputType, Dims>;

    for (int i = 0; i < input.shape().num_samples(); i++) {
      thread_pool.AddWork(
        [this, &input, &output, i](int thread_id) {
          kernels::KernelContext ctx;
          auto in_view = view<const InputType, Dims>(input[i]);
          auto out_view = view<OutputType, Dims>(output[i]);
          kmgr_.Run<FftKernel>(thread_id, i, ctx, out_view, in_view, fft_args_);
        }, in_shape.tensor_size(i));
    }
  ), DALI_FAIL(make_string("Not supported number of dimensions: ", in_shape.size())));  // NOLINT

  thread_pool.RunAll();
}

DALI_REGISTER_OPERATOR(PowerSpectrum, PowerSpectrum<CPUBackend>, CPU);

}  // namespace dali
