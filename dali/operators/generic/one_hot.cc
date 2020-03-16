// Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/generic/one_hot.h"
#include <iostream>

namespace dali {

DALI_REGISTER_OPERATOR(OneHot, OneHot, CPU);

DALI_SCHEMA(OneHot)
    .DocStr(
        "Produce tensor representing one hot encoding "
        " of the given input")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("depth", R"code(Number of all classes)code", 0)
    .AddOptionalArg(arg_names::kDtype, R"code(Data type for the output)code",
                    DALI_FLOAT)
    .AddOptionalArg("on_value", R"code(Value that will be used to fill the output when input[j] = i)code", 1)
    .AddOptionalArg("off_value", R"code(Value that will be used to fill the output when input[j] != i)code", 0);

bool OneHot::SetupImpl(std::vector<OutputDesc> &output_desc, const HostWorkspace &ws) {
  output_desc.resize(1);
  output_desc[0].shape = uniform_list_shape(batch_size_, {depth_});
  TYPE_SWITCH(output_type_, type2id, DType, ONE_HOT_TYPES, (
    {
      TypeInfo type;
      type.SetType<DType>(output_type_);
      output_desc[0].type = type;
    }
  ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  return true;
}

void OneHot::RunImpl(HostWorkspace &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  auto &output = ws.template OutputRef<CPUBackend>(0);
  auto &tp = ws.GetThreadPool();
  TYPE_SWITCH(input.type().id(), type2id, InputType, ONE_HOT_TYPES, (
    TYPE_SWITCH(output_type_, type2id, OutputType, ONE_HOT_TYPES, (
      for (int sample_id = 0; sample_id < batch_size_; ++sample_id) {
        tp.DoWorkWithID(
              [&, sample_id](int thread_id) {
          auto &in = input[sample_id];
          auto &out = output[sample_id];
          auto in_tensor = make_tensor_cpu(in.template data<InputType>(), in.shape());
          auto out_tensor = make_tensor_cpu(out.template mutable_data<OutputType>(), out.shape());
          detail::DoOneHot<OutputType, InputType>(out_tensor, 
                                                  in_tensor, 
                                                  depth_, 
                                                  on_value_, 
                                                  off_value_);
        });
      }
    ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())))  // NOLINT
  tp.WaitForWork();
}

}  // namespace dali