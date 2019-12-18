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

#include "dali/operators/audio/mel_scale/mel_filter_bank.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/audio/mel_scale/mel_filter_bank_cpu.h"
#include "dali/pipeline/data/views.h"

#define MEL_FBANK_SUPPORTED_TYPES (float)
#define MEL_FBANK_SUPPORTED_NDIMS (2, 3, 4)

static constexpr int kNumInputs = 1;
static constexpr int kNumOutputs = 1;

namespace dali {

DALI_SCHEMA(MelFilterBank)
    .DocStr(R"code(Converts a Spectrogram to a mel Spectrogram using triangular filter banks.
Expects an input with 2 or more dimensions where the last two dimensions correspond to the
fft bin index and the window index respectively.)code")
    .NumInput(kNumInputs)
    .NumOutput(kNumOutputs)
    .AddOptionalArg("nfilter",
      R"code(Number of mel filters.)code",
      128)
    .AddOptionalArg("sample_rate",
      R"code(Sampling rate of the audio signal)code",
      44100.0f)
    .AddOptionalArg("freq_low",
      R"code(Minimum frequency)code",
      0.0f)
    .AddOptionalArg("freq_high",
      R"code(Maximum frequency. If not provided, `sample_rate / 2` will be used)code",
      0.0f)
    .AddOptionalArg("normalize",
      R"code(Whether to normalize the triangular filter weights by the width of their mel band.
If set to true, the integral of the filter function will amount to 1.
If set to false, the peak of the filter function will be 1)code",
      true)
    .AddOptionalArg("mel_formula",
      R"code(Determines the formula used to convert frequencies from Hertz to mel and viceversa.
The mel scale is a perceptual scale of pitches and therefore there is no single formula to it.
Supported values are:
- \"slaney\" : Follows Slaney's MATLAB Auditory Modelling Work behavior. This formula is linear
under 1 KHz and logarithmic above. This implementation is consistent with Librosa's default
implementation.
- \"htk\" : Follows O'Shaughnessy's book formula `m = 2595 * log10(1 + (f/700))`. This is
consistent with the implementation of the Hidden Markov Toolkit (HTK).
)code",
      "slaney");

template <>
bool MelFilterBank<CPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                          const workspace_t<CPUBackend> &ws) {
  output_desc.resize(kNumOutputs);
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  kernels::KernelContext ctx;
  auto in_shape = input.shape();
  int nsamples = input.size();
  auto nthreads = ws.GetThreadPool().size();

  TYPE_SWITCH(input.type().id(), type2id, T, MEL_FBANK_SUPPORTED_TYPES, (
    VALUE_SWITCH(in_shape.sample_dim(), Dims, MEL_FBANK_SUPPORTED_NDIMS, (
      using MelFilterBankKernel = kernels::audio::MelFilterBankCpu<T, Dims>;
      kmgr_.Initialize<MelFilterBankKernel>();
      kmgr_.Resize<MelFilterBankKernel>(nthreads, nsamples);
      output_desc[0].type = TypeInfo::Create<T>();
      output_desc[0].shape.resize(nsamples, Dims);
      for (int i = 0; i < nsamples; i++) {
        const auto in_view = view<const T, Dims>(input[i]);
        auto &req = kmgr_.Setup<MelFilterBankKernel>(i, ctx, in_view, args_);
        output_desc[0].shape.set_tensor_shape(i, req.output_shapes[0][0].shape);
      }
    ), DALI_FAIL(make_string("Unsupported number of dimensions ", in_shape.size())));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type().id())));  // NOLINT
  return true;
}

template <>
void MelFilterBank<CPUBackend>::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto in_shape = input.shape();
  int nsamples = input.size();
  auto& thread_pool = ws.GetThreadPool();

  TYPE_SWITCH(input.type().id(), type2id, T, MEL_FBANK_SUPPORTED_TYPES, (
    VALUE_SWITCH(in_shape.sample_dim(), Dims, MEL_FBANK_SUPPORTED_NDIMS, (
      using MelFilterBankKernel = kernels::audio::MelFilterBankCpu<T, Dims>;
      for (int i = 0; i < input.shape().num_samples(); i++) {
        thread_pool.DoWorkWithID(
          [this, &input, &output, i](int thread_id) {
            kernels::KernelContext ctx;
            auto in_view = view<const T, Dims>(input[i]);
            auto out_view = view<T, Dims>(output[i]);
            kmgr_.Run<MelFilterBankKernel>(thread_id, i, ctx, out_view, in_view, args_);
          });
      }
    ), DALI_FAIL(make_string("Unsupported number of dimensions ", in_shape.size())));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type().id())));  // NOLINT

  thread_pool.WaitForWork();
}

DALI_REGISTER_OPERATOR(MelFilterBank, MelFilterBank<CPUBackend>, CPU);

}  // namespace dali
