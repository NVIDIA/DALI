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

namespace dali {

DALI_SCHEMA(MelFilterBank)
    .DocStr(R"code(Converts a spectrogram to a mel spectrogram by applying a bank of
triangular filters.

The frequency ('f') dimension is selected from the input layout.
In case of no layout, "f", "ft", or "*ft" is assumed, depending on the number of dimensions.
)code")
    .NumInput(kNumInputs)
    .NumOutput(kNumOutputs)
    .AddOptionalArg("nfilter",
      R"code(Number of mel filters.)code",
      128)
    .AddOptionalArg("sample_rate",
      R"code(Sampling rate of the audio signal.)code",
      44100.0f)
    .AddOptionalArg("freq_low",
      R"code(The minimum frequency.)code",
      0.0f)
    .AddOptionalArg("freq_high",
      R"code(The maximum frequency.

If this value is not provided, ``sample_rate /2`` is used.
)code",
      0.0f)
    .AddOptionalArg("normalize",
      R"code(Determines whether to normalize the triangular filter weights by the width
of their frequency bands.

- If set to True, the integral of the filter function is 1.
- If set to False, the peak of the filter function will be 1.)code",
      true)
    .AddOptionalArg("mel_formula",
      R"code(Determines the formula that will be used to convert frequencies from hertz to mel
and from mel to hertz.

The mel scale is a perceptual scale of pitches, so there is no single formula.

The supported values are:

- | ``slaney``, which follows Slaney's MATLAB Auditory Modelling Work behavior.
  | This formula is linear under 1 KHz and logarithmic above this value. The implementation is
    consistent with Librosa's default implementation.
- | ``htk``, which follows O'Shaughnessy's book formula, ``m = 2595 * log10(1 + (f/700))``.
  | This value is consistent with the implementation of the Hidden Markov Toolkit (HTK).
)code", "slaney");

template <>
bool MelFilterBank<CPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                          const workspace_t<CPUBackend> &ws) {
  output_desc.resize(kNumOutputs);
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto in_shape = input.shape();
  int nsamples = input.size();
  auto nthreads = ws.GetThreadPool().NumThreads();
  auto layout = input.GetLayout();
  auto ndim = in_shape.sample_dim();
  args_.axis = layout.empty() ? std::max(0, ndim - 2) : layout.find('f');
  DALI_ENFORCE(args_.axis >= 0 && args_.axis < ndim,
    make_string("'f' axis not present in the layout. Got: `", layout, "`"));
  TYPE_SWITCH(input.type().id(), type2id, T, MEL_FBANK_SUPPORTED_TYPES, (
    using MelFilterBankKernel = kernels::audio::MelFilterBankCpu<T>;
    kmgr_.Initialize<MelFilterBankKernel>();
    kmgr_.Resize<MelFilterBankKernel>(nthreads, nsamples);
    output_desc[0].type = TypeTable::GetTypeInfo(TypeTable::GetTypeID<T>());
    const auto in_view = view<const T>(input);
    output_desc[0].shape.resize(nsamples, in_view.sample_dim());
    for (int i = 0; i < nsamples; i++) {
      auto &req = kmgr_.Setup<MelFilterBankKernel>(i, ctx_, in_view[i], args_);
      output_desc[0].shape.set_tensor_shape(i, req.output_shapes[0][0].shape);
    }
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type().id())));  // NOLINT
  return true;
}

template <>
void MelFilterBank<CPUBackend>::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto in_shape = input.shape();
  auto& thread_pool = ws.GetThreadPool();

  TYPE_SWITCH(input.type().id(), type2id, T, MEL_FBANK_SUPPORTED_TYPES, (
    using MelFilterBankKernel = kernels::audio::MelFilterBankCpu<T>;
    for (int i = 0; i < input.shape().num_samples(); i++) {
      thread_pool.AddWork(
        [this, &input, &output, i](int thread_id) {
          auto in_view = view<const T>(input[i]);
          auto out_view = view<T>(output[i]);
          kmgr_.Run<MelFilterBankKernel>(thread_id, i, ctx_, out_view, in_view);
        }, in_shape.tensor_size(i));
    }
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type().id())));  // NOLINT

  thread_pool.RunAll();
}

DALI_REGISTER_OPERATOR(MelFilterBank, MelFilterBank<CPUBackend>, CPU);

}  // namespace dali
