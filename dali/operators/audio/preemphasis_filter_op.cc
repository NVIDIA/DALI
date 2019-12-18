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
#include "dali/core/static_switch.h"

namespace dali {


DALI_SCHEMA(PreemphasisFilter)
                .DocStr(R"code(This operator performs preemphasis filter on the input data.
This filter in simple form can be expressed by the formula::

  Y(t) = X(t) - X(t-1)*coeff)code")
                .NumInput(1)
                .NumOutput(detail::kNumOutputs)
                .AddOptionalArg(detail::kCoeff, R"code(Preemphasis coefficient `coeff`)code", 0.97f)
                .AddOptionalArg(arg_names::kDtype, R"code(Data type for the output)code",
                                DALI_FLOAT);

DALI_REGISTER_OPERATOR(PreemphasisFilter, PreemphasisFilterCpu, CPU);

#define PREEMPH_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float, double)  // NOLINT


bool PreemphasisFilterCpu::SetupImpl(std::vector <OutputDesc> &output_desc,
                                     const workspace_t <CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  AcquireArguments(ws);
  output_desc.resize(detail::kNumOutputs);
  output_desc[0].shape = input.shape();
  TYPE_SWITCH(output_type_, type2id, DType, PREEMPH_TYPES, (
          {
            TypeInfo type;
            type.SetType<DType>(output_type_);
            output_desc[0].type = type;
          }
  ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  return true;
}


void PreemphasisFilterCpu::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto &tp = ws.GetThreadPool();
  TYPE_SWITCH(input.type().id(), type2id, InputType, PREEMPH_TYPES, (
    TYPE_SWITCH(output_type_, type2id, OutputType, PREEMPH_TYPES, (
          for (int sample_id = 0; sample_id < batch_size_; ++sample_id) {
            tp.DoWorkWithID(
              [&, sample_id](int thread_id) {
                const auto in_ptr = input[sample_id].data<InputType>();
                auto out_ptr = output[sample_id].mutable_data<OutputType>();
                auto num_samples = volume(output[sample_id].shape());
                DALI_ENFORCE(input[sample_id].shape() == output[sample_id].shape(),
                          "Input and output shapes don't match");
                if (preemph_coeff_[sample_id] == 0.f) {
                  for (long j = 0; j < num_samples; j++) {  // NOLINT (long)
                    out_ptr[j] = ConvertSat<OutputType>(in_ptr[j]);
                  }
                } else {
                  for (auto j = num_samples - 1; j > 0; j--) {
                    out_ptr[j] = ConvertSat<OutputType>(
                            in_ptr[j] - in_ptr[j - 1] * preemph_coeff_[sample_id]);
                  }
                  out_ptr[0] = ConvertSat<OutputType>(in_ptr[0] * preemph_coeff_[sample_id]);
                }
              });
          }
    ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())))  // NOLINT
  tp.WaitForWork();
}

#undef PREEMPH_TYPES

}  // namespace dali
