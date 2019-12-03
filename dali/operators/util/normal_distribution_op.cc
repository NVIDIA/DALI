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
This operator can be ran in 3 modes, which determine the shape of the output tensor:
1. Providing an input batch to this operator resolves in a batch of output tensors,
where the shapes correspond to the shapes of tensors in the input batch.
2. Providing a custom shape as an argument resolves in a batch of output tensors,
where every output tensor has the same shape.
3. Providing no input arguments resolves in a batch of output scalars,
distributed normally (similar to `CoinFlip` operator).)code")
                .NumInput(0, 1)
                .NumOutput(detail::kNumOutputs)
                .AddOptionalArg(detail::kMean, R"code(Mean value of the distribution)code",
                                0.f, true)
                .AddOptionalArg(detail::kStddev,
                                R"code(Standard deviation of the distribution)code",
                                1.f, true)
                .AddOptionalArg<float>(detail::kShape,
                                R"code(Shape of single output tensor in a batch)code",
                                       detail::kShapeDefaultValue)
                .AddOptionalArg(arg_names::kDtype, R"code(Data type for the output)code",
                                DALI_FLOAT);

DALI_REGISTER_OPERATOR(NormalDistribution, NormalDistributionCpu, CPU);


#define NORM_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float, double)  // NOLINT


bool NormalDistributionCpu::SetupImpl(std::vector<OutputDesc> &output_desc,
                                      const workspace_t<CPUBackend> &ws) {
  cout<<__PRETTY_FUNCTION__<<endl<<endl;
  AcquireArguments(ws);
  output_desc.resize(detail::kNumOutputs);
  output_desc[0].shape = GetOutputShape(ws);
  cout<<"SHAPE "<<output_desc[0].shape<<endl;
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
  cout<<__PRETTY_FUNCTION__<<endl<<endl;
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto &tp = ws.GetThreadPool();
  distribution_t distribution(mean_[0], stddev_[0]);
  TYPE_SWITCH(dtype_, type2id, DType, NORM_TYPES, (
          for (int sample_id = 0; sample_id < batch_size_; ++sample_id) {
          tp.DoWorkWithID([&, sample_id](int thread_id){
//            cout<<"DUPA "<<sample_id<<endl;
              auto ptr = output[sample_id].mutable_data<DType>();
//            cout<<"DUPA "<<sample_id<<endl;
            *ptr = ConvertSat<DType>(distribution(rng_));
            cout<<"DUPA "<<sample_id<<" "<<*ptr<<endl;
          });
  }
  ), DALI_FAIL(make_string("Unsupported output type: ", dtype_)))  // NOLINT
  tp.WaitForWork();
}


void NormalDistributionCpu::AssignTensorToOutput(workspace_t<CPUBackend> &ws) {
  cout<<__PRETTY_FUNCTION__<<endl<<endl;
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto &tp = ws.GetThreadPool();
  TYPE_SWITCH(dtype_, type2id, DType, NORM_TYPES, (
            for (int sample_id = 0; sample_id < batch_size_; ++sample_id) {
                distribution_t distribution(mean_[sample_id], stddev_[sample_id]);
                auto ptr = output[sample_id].mutable_data<DType>();
                for (long j = 0; j < volume(output[sample_id].shape()); j++) {  // NOLINT (long)
                    ptr[j] = ConvertSat<DType>(distribution(rng_));
                }
            }
  ), DALI_FAIL(make_string("Unsupported output type: ", dtype_)))  // NOLINT
}


void NormalDistributionCpu::RunImpl(workspace_t<CPUBackend> &ws) {
  cout<<__PRETTY_FUNCTION__<<endl<<endl;
  if (this->single_value_in_output_) {
    AssignSingleValueToOutput(ws);
  } else {
    AssignTensorToOutput(ws);
  }
}


#undef NORM_TYPES

}  // namespace dali
