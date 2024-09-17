// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/generic/constant_value.h"
#include <vector>
#include "dali/core/convert.h"
#include "dali/pipeline/operator/op_schema.h"

namespace dali {

template <>
void ConstantValue<CPUBackend>::RunImpl(Workspace &ws) {
  auto &output = ws.Output<CPUBackend>(0);
  const auto& out_shape = output.shape();
  int nsamples = out_shape.size();
  auto dtype = output.type();
  auto &tp = ws.GetThreadPool();
  if (has_fill_value_) {
    auto &fill_value = ws.Input<CPUBackend>(value_input_idx_);
    const auto &fill_value_sh = fill_value.shape();
    TYPE_SWITCH(fill_value.type(), type2id, FillValueType, (DALI_CONSTANT_VALUE_TYPES), (
      TYPE_SWITCH(dtype, type2id, OutputType, (DALI_CONSTANT_VALUE_TYPES), (
        for (int s = 0; s < nsamples; s++) {
          tp.AddWork([&, s](int thread_idx) {
            const auto* fill_value_data = fill_value.tensor<FillValueType>(s);
            auto* out_data = output.mutable_tensor<OutputType>(s);
            auto out_sz = out_shape.tensor_size(s);
            auto fill_value_sz = fill_value_sh.tensor_size(s);
            for (int64_t i = 0; i < out_sz; i++) {
              out_data[i] = ConvertSat<OutputType>(fill_value_data[i % fill_value_sz]);
            }
          });
        }
        tp.RunAll();
      ), (  // NOLINT
        DALI_FAIL(
          make_string("Data type ", dtype, " is currently not supported. "
                      "Supported types are : ", ListTypeNames<DALI_CONSTANT_VALUE_TYPES>()));
      ));  // NOLINT
    ), (  // NOLINT
      DALI_FAIL(
        make_string("Data type ", fill_value.type(), " is currently not supported. "
                    "Supported types are : ", ListTypeNames<DALI_CONSTANT_VALUE_TYPES>()));
    ));  // NOLINT
  } else {
    TYPE_SWITCH(dtype, type2id, T, (DALI_CONSTANT_VALUE_TYPES), (
      T value = ConvertSat<T>(const_value_);
      for (int s = 0; s < nsamples; s++) {
        tp.AddWork([&, value, s](int thread_idx) {
          auto* out_data = output.mutable_tensor<T>(s);
          auto out_sz = out_shape.tensor_size(s);
          std::fill(out_data, out_data + out_sz, value);
        });
      }
      tp.RunAll();
    ), (  // NOLINT
      DALI_FAIL(make_string("Data type ", dtype, " is currently not supported. "
                            "Supported types are : ", ListTypeNames<DALI_CONSTANT_VALUE_TYPES>()));
    ));  // NOLINT
  }
}

DALI_SCHEMA(Full)
    .DocStr(R"code(Returns new data of given shape and type, filled with a fill value.)code")
    .NumInput(1)
    .InputDox(0, "fill_value", "TensorList", R"code(The fill value.)code")
    .NumOutput(1)
    .AddOptionalArg<std::vector<int>>("shape", R"code(Shape of the output data.)code", nullptr,
                                      true);
DALI_REGISTER_OPERATOR(Full, Full<CPUBackend>, CPU);

DALI_SCHEMA(FullLike)
    .DocStr(R"code(Returns new data with the same shape and type as the input data, filled with a `fill_value`.)code")
    .NumInput(2)
    .InputDox(0, "data_like", "TensorList", R"code(The input data value to copy the shape and type from.)code")
    .InputDevice(0, InputDevice::Metadata)
    .InputDox(1, "fill_value", "TensorList", R"code(The fill value.)code")
    .NumOutput(1);
DALI_REGISTER_OPERATOR(FullLike, FullLike<CPUBackend>, CPU);

DALI_SCHEMA(Zeros)
    .DocStr(R"code(Returns new data of given shape and type, filled with zeros.)code")
    .NumInput(0)
    .NumOutput(1)
    .AddOptionalArg<std::vector<int>>("shape", R"code(Shape of the output data.)code", nullptr,
                                      true)
    .AddOptionalTypeArg("dtype", R"code(Output data type.)code", DALI_INT32);
DALI_REGISTER_OPERATOR(Zeros, Zeros<CPUBackend>, CPU);

DALI_SCHEMA(ZerosLike)
    .DocStr(R"code(Returns new data with the same shape and type as the input array, filled with zeros.)code")
    .NumInput(1)
    .InputDox(0, "data_like", "TensorList", R"code(The input data value to copy the shape and type from.)code")
    .InputDevice(0, InputDevice::Metadata)
    .NumOutput(1)
    .AddOptionalTypeArg("dtype", R"code(Overrides the output data type.)code", DALI_INT32);
DALI_REGISTER_OPERATOR(ZerosLike, ZerosLike<CPUBackend>, CPU);

DALI_SCHEMA(Ones)
    .DocStr(R"code(Returns new data of given shape and type, filled with ones.)code")
    .NumInput(0)
    .NumOutput(1)
    .AddOptionalArg<std::vector<int>>("shape", R"code(Shape of the output data.)code", nullptr,
                                      true)
    .AddOptionalTypeArg("dtype", R"code(Output data type.)code", DALI_INT32);
DALI_REGISTER_OPERATOR(Ones, Ones<CPUBackend>, CPU);

DALI_SCHEMA(OnesLike)
    .DocStr(R"code(Returns new data with the same shape and type as the input array, filled with ones.)code")
    .NumInput(1)
    .InputDox(0, "data_like", "TensorList", R"code(The input data value to copy the shape and type from.)code")
    .InputDevice(0, InputDevice::Metadata)
    .NumOutput(1)
    .AddOptionalTypeArg("dtype", R"code(Overrides the output data type.)code", DALI_INT32);
DALI_REGISTER_OPERATOR(OnesLike, OnesLike<CPUBackend>, CPU);

}  // namespace dali
