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

namespace dali {

DALI_SCHEMA(NormalDistribution)
    .DocStr(R"code(Creates a batch of tensors filled with random values following a normal
distribution.

This operator can be run in the following modes, which determine the ``shape`` of the output
tensors/batch:

* Providing an input batch to this operator results in a batch of output tensors, which have
  the same ``shape`` as the input tensors.
* Providing a custom ``shape`` as an argument results in an output batch, where every tensor has
  the same (given) ``shape``.
* Providing no input arguments results in an output batch of scalars, distributed normally.)code")
  .NumInput(0, 1)
  .NumOutput(detail::kNumOutputs)
  .AddOptionalArg(detail::kMean, R"code(Mean value of the distribution.)code",
                  0.f, true)
  .AddOptionalArg(detail::kStddev,
                  R"code(Standard deviation of the distribution.)code",
                  1.f, true)
  .AddOptionalArg(detail::kShape,
                  R"code(Shape of an output tensor in a batch.)code",
                  detail::kShapeDefaultValue)
  .AddOptionalArg(arg_names::kDtype, R"code(Output data type.)code",
                  DALI_FLOAT);

DALI_REGISTER_OPERATOR(NormalDistribution, NormalDistributionCpu, CPU);

void NormalDistributionCpu::AssignSingleValueToOutput(workspace_t<CPUBackend> &ws) {
  auto &output = ws.OutputRef<CPUBackend>(0);
  distribution_t distribution(mean_[0], stddev_[0]);
  TYPE_SWITCH(dtype_, type2id, DType, DALI_NORMDIST_TYPES, (
          for (int sample_id = 0; sample_id < max_batch_size_; ++sample_id) {
            auto ptr = output[sample_id].mutable_data<DType>();
            *ptr = ConvertSat<DType>(distribution(rng_));
          }
  ), DALI_FAIL(make_string("Unsupported output type: ", dtype_)))  // NOLINT
}


void NormalDistributionCpu::AssignTensorToOutput(workspace_t<CPUBackend> &ws) {
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto out_shape = output.shape();
  auto &tp = ws.GetThreadPool();
  TYPE_SWITCH(dtype_, type2id, DType, DALI_NORMDIST_TYPES, (
            for (int sample_id = 0; sample_id < max_batch_size_; ++sample_id) {
              auto out_size = out_shape.tensor_size(sample_id);
              tp.AddWork(
                  [&, sample_id, out_size](int thread_id) {
                     distribution_t distribution(mean_[sample_id], stddev_[sample_id]);
                     auto ptr = output[sample_id].mutable_data<DType>();
                     for (int64_t j = 0; j < out_size; j++) {
                         ptr[j] = ConvertSat<DType>(distribution(batch_rng_[sample_id]));
                     }
                  }, out_size);
            }
  ), DALI_FAIL(make_string("Unsupported output type: ", dtype_)))  // NOLINT
  tp.RunAll();
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
