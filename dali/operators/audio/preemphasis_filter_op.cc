// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
This filter in simple form can be expressed by the formula:
`Y(t) = X(t) - X(t-1)*coeff`)code")
                .NumInput(1)
                .NumOutput(detail::kNumOutputs)
                .AddOptionalArg(detail::kCoeff, R"code(Preemphasis coefficient `coeff`)code", 0.f)
                .AddOptionalArg(detail::kDtype, R"code(Data type for the output)code", DALI_FLOAT);

DALI_REGISTER_OPERATOR(PreemphasisFilter, PreemphasisFilterCpu, CPU);


#define PREEMPH_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float, double)  // NOLINT


bool PreemphasisFilterCpu::SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                                     const workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  output_desc.resize(detail::kNumOutputs);
  output_desc[0].shape = input.shape();
  TYPE_SWITCH(dtype_, type2id, DType, PREEMPH_TYPES, (
          {
            TypeInfo type;
            type.SetType<DType>(dtype_);
            output_desc[0].type = type;
          }
  ), DALI_FAIL(make_string("Unsupported output type: ", dtype_)))  // NOLINT
  return true;
}


void PreemphasisFilterCpu::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  for (int i = 0; i < batch_size_; ++i) {
    TYPE_SWITCH(dtype_, type2id, DType, PREEMPH_TYPES, (
            {
              const auto in_ptr = input[i].data<DType>();
              auto out_ptr = output[i].mutable_data<DType>();
              auto num_samples = volume(output[i].shape());
              DALI_ENFORCE(input[i].shape() == output[i].shape(),
                           "Input and output shapes don't match");
              if (preemph_coeff_ == 0.f) {
                for (int j = 0; j < num_samples; j++) {
                  out_ptr[j] = ConvertSat<DType>(in_ptr[j]);
                }
              } else {
                for (auto j = num_samples - 1; j > 0; j--) {
                  out_ptr[j] = ConvertSat<DType>(in_ptr[j] - in_ptr[j - 1] * preemph_coeff_);
                }
                out_ptr[0] = ConvertSat<DType>(in_ptr[0] * preemph_coeff_);
              }
            }
    ), DALI_FAIL(make_string("Unsupported output type: ", dtype_)))  // NOLINT
  }
}


#undef PREEMPH_TYPES

}  // namespace dali
