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

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include "dali/operators/audio/preemphasis_filter_op.h"

namespace dali {

DALI_SCHEMA(PreemphasisFilter)
    .DocStr(R"code(Applies preemphasis filter to the input data.

This filter, in simple form, can be expressed by the formula::

  Y[t] = X[t] - coeff * X[t-1]        if t > 1

The behavior for the best sample, depends on the ``reflect_padding`` argument::

  Y[0] = X[0]                         if reflect_padding == True
  Y[0] = X[0] - coeff * X[0]          otherwise

Where:

  - ``X`` is the input singal.
  - ``Y`` is the output signal.

)code")
    .NumInput(1)
    .NumOutput(detail::kNumOutputs)
    .AddOptionalArg(detail::kCoeff, R"code(Preemphasis coefficient ``coeff``.)code", 0.97f, true)
    .AddOptionalArg(arg_names::kDtype, R"code(Data type for the output.)code", DALI_FLOAT)
    .AddOptionalArg(detail::kReflectPadding,
      R"(Indicates the padding policy when sampling outside the bounds of the signal (first sample).

If True, the signal is mirrored, otherwise the signal is padded with zeros.)",
    true);

class PreemphasisFilterCPU : public PreemphasisFilter<CPUBackend> {
 public:
  explicit PreemphasisFilterCPU(const OpSpec &spec) : PreemphasisFilter<CPUBackend>(spec) {}
  void RunImpl(workspace_t<CPUBackend> &ws) override;

 private:
  template <typename OutputType, typename InputType>
  void RunImplTyped(workspace_t<CPUBackend> &ws);
};

template <typename OutputType, typename InputType>
void PreemphasisFilterCPU::RunImplTyped(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto &tp = ws.GetThreadPool();
  auto shape = input.shape();
  auto nsamples = shape.num_samples();

  for (int sample_id = 0; sample_id < nsamples; sample_id++) {
    tp.AddWork(
      [this, &output, &input, sample_id](int thread_id) {
        const auto in_ptr = input[sample_id].data<InputType>();
        auto out_ptr = output[sample_id].mutable_data<OutputType>();
        DALI_ENFORCE(input[sample_id].shape() == output[sample_id].shape(),
                     "Input and output shapes don't match");
        auto n = volume(output[sample_id].shape());
        auto coeff = preemph_coeff_[sample_id];
        if (coeff == 0.0f) {
          for (int64_t j = 0; j < n; j++) {
            out_ptr[j] = ConvertSat<OutputType>(in_ptr[j]);
          }
        } else {
          InputType in_prev = reflect_padding_ ? in_ptr[0] : 0;
          out_ptr[0] = ConvertSat<OutputType>(in_ptr[0] - coeff * in_prev);
          for (int64_t j = 1; j < n; j++) {
            out_ptr[j] = ConvertSat<OutputType>(in_ptr[j] - coeff * in_ptr[j - 1]);
          }
        }
      }, shape.tensor_size(sample_id));
  }
  tp.RunAll();
}

void PreemphasisFilterCPU::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  TYPE_SWITCH(input.type().id(), type2id, InputType, PREEMPH_TYPES, (
    TYPE_SWITCH(output_type_, type2id, OutputType, PREEMPH_TYPES, (
      RunImplTyped<OutputType, InputType>(ws);
    ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())));  // NOLINT
}

DALI_REGISTER_OPERATOR(PreemphasisFilter, PreemphasisFilterCPU, CPU);

}  // namespace dali
