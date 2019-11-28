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

#include "dali/operators/util/normal_distribution_op.h"
#include "dali/core/static_switch.h"

namespace dali {


DALI_SCHEMA(NormalDistribution)
                .DocStr(R"code(Creates a tensor that consists of data distributed normally.
The shape of the provided tensor is implied by the tensor provided as an input.
No data from the input tensor is taken into account, only the shape of it.)code")
                .NumInput(1)
                .NumOutput(detail::kNumOutputs)
                .AddOptionalArg(detail::kMean, R"code(Mean value of the distribution)code", 0.f)
                .AddOptionalArg(detail::kStddev,
                                R"code(Standard deviation of the distribution)code", 1.f)
                .AddOptionalArg(detail::kDtype, R"code(Data type for the output)code", DALI_FLOAT);

DALI_REGISTER_OPERATOR(NormalDistribution, NormalDistributionCpu, CPU);


#define NORM_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float, double)  // NOLINT


bool NormalDistributionCpu::SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                                      const workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  output_desc.resize(detail::kNumOutputs);
  output_desc[0].shape = input.shape();
  TYPE_SWITCH(dtype_, type2id, DType, NORM_TYPES, (
          {
            TypeInfo type;
            type.SetType<DType>(dtype_);
            output_desc[0].type = type;
          }
  ), DALI_FAIL(make_string("Unsupported output type: ", dtype_)))  // NOLINT
  return true;
}


void NormalDistributionCpu::RunImpl(workspace_t<CPUBackend> &ws) {
  auto &output = ws.OutputRef<CPUBackend>(0);
  for (int i = 0; i < batch_size_; ++i) {
    TYPE_SWITCH(dtype_, type2id, DType, NORM_TYPES, (
            {
              auto ptr = output[i].mutable_data<DType>();
              for (long j = 0; j < volume(output[i].shape()); j++) {  // NOLINT (long)
                ptr[j] = ConvertSat<DType>(distribution_(rng_));
              }
            }
    ), DALI_FAIL(make_string("Unsupported output type: ", dtype_)))  // NOLINT
  }
}


#undef NORM_TYPES

}  // namespace dali
