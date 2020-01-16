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

#include "dali/operators/random/normal_distribution_op.h"
#include "dali/core/static_switch.h"

#define NORM_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, \
                    float16, float, double)

namespace dali {

DALI_SCHEMA(NormalDistribution)
                .DocStr(R"code(Creates a tensor that consists of data distributed normally.
This operator can be ran in 3 modes, which determine the shape of the output tensor:
1. Providing an input batch to this operator results in a batch of output tensors, which have the same shape as the input tensors.
2. Providing a custom `shape` as an argument results in an output batch, where every tensor has the same (given) shape.
3. Providing no input arguments results in an output batch of scalars, distributed normally.)code")
                .NumInput(0, 1)
                .NumOutput(detail::kNumOutputs)
                .AddOptionalArg(detail::kMean, R"code(Mean value of the distribution)code",
                                0.f, true)
                .AddOptionalArg(detail::kStddev,
                                R"code(Standard deviation of the distribution)code",
                                1.f, true)
                .AddOptionalArg(detail::kShape,
                                R"code(Shape of single output tensor in a batch)code",
                                detail::kShapeDefaultValue)
                .AddOptionalArg(arg_names::kDtype, R"code(Data type for the output)code",
                                DALI_FLOAT);

DALI_REGISTER_OPERATOR(NormalDistribution, NormalDistributionCpu, CPU);

bool NormalDistributionCpu::SetupImpl(std::vector<OutputDesc> &output_desc,
                                      const workspace_t<CPUBackend> &ws) {
  AcquireArguments(ws);
  output_desc.resize(detail::kNumOutputs);
  output_desc[0].shape = GetOutputShape(ws);
  TYPE_SWITCH(dtype_, type2id, DType, NORM_TYPES, (
          {
            TypeInfo type;
            type.SetType<DType>(dtype_);
            output_desc[0].type = type;
          }
  ), DALI_FAIL(make_string("Unsupported output type: ", dtype_)))  // NOLINT
  return true;
}


void NormalDistributionCpu::AssignSingleValueToOutput(workspace_t<CPUBackend> &ws) {
  auto &output = ws.OutputRef<CPUBackend>(0);
  distribution_t distribution(mean_[0], stddev_[0]);
  TYPE_SWITCH(dtype_, type2id, DType, NORM_TYPES, (
          for (int sample_id = 0; sample_id < batch_size_; ++sample_id) {
            auto ptr = output[sample_id].mutable_data<DType>();
            *ptr = ConvertSat<DType>(distribution(rng_));
          }
  ), DALI_FAIL(make_string("Unsupported output type: ", dtype_)))  // NOLINT
}


void NormalDistributionCpu::AssignTensorToOutput(workspace_t<CPUBackend> &ws) {
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto &tp = ws.GetThreadPool();
  TYPE_SWITCH(dtype_, type2id, DType, NORM_TYPES, (
            for (int sample_id = 0; sample_id < batch_size_; ++sample_id) {
              tp.DoWorkWithID(
                  [&, sample_id](int thread_id) {
                     distribution_t distribution(mean_[sample_id], stddev_[sample_id]);
                     auto ptr = output[sample_id].mutable_data<DType>();
                     for (int64_t j = 0; j < volume(output[sample_id].shape()); j++) {
                         ptr[j] = ConvertSat<DType>(distribution(batch_rng_[sample_id]));
                     }
                  });
            }
  ), DALI_FAIL(make_string("Unsupported output type: ", dtype_)))  // NOLINT
  tp.WaitForWork();
}


void NormalDistributionCpu::RunImpl(workspace_t<CPUBackend> &ws) {
  if (this->single_value_in_output_) {
    AssignSingleValueToOutput(ws);
  } else {
    AssignTensorToOutput(ws);
  }
}


#undef NORM_TYPES

}  // namespace dali
