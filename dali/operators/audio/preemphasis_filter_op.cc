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

#include "dali/operators/audio/preemphasis_filter_op.h"

namespace dali {

DALI_SCHEMA(PreemphasisFilter)
    .DocStr(R"code(This operator performs preemphasis filter on the input data.
This filter in simple form can be expressed by the formula::

  Y(t) = X[t] - coeff * X[t-1])code")
    .NumInput(1)
    .NumOutput(detail::kNumOutputs)
    .AddOptionalArg(detail::kCoeff, R"code(Preemphasis coefficient `coeff`)code", 0.97f, true)
    .AddOptionalArg(arg_names::kDtype, R"code(Data type for the output)code", DALI_FLOAT);

template <>
void PreemphasisFilter<CPUBackend>::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto &tp = ws.GetThreadPool();
  TYPE_SWITCH(input.type().id(), type2id, InputType, PREEMPH_TYPES, (
    TYPE_SWITCH(output_type_, type2id, OutputType, PREEMPH_TYPES, (
      while (!sample_queue_.empty()) {
        auto sample_id = sample_queue_.top().second;
        sample_queue_.pop();
        tp.DoWorkWithID(
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
              out_ptr[0] = ConvertSat<OutputType>(in_ptr[0] - coeff * in_ptr[0]);
              for (int64_t j = 1; j < n; j++) {
                out_ptr[j] = ConvertSat<OutputType>(in_ptr[j] - coeff * in_ptr[j - 1]);
              }
            }
          });
      }
    ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())));  // NOLINT
  tp.WaitForWork();
}

DALI_REGISTER_OPERATOR(PreemphasisFilter, PreemphasisFilter<CPUBackend>, CPU);

}  // namespace dali
