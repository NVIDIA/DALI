// Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/kernels/common/utils.h"

namespace dali {

template <typename T, typename U>
void RepeatInner(T *out, int64_t out_size, const U *in, int64_t in_size) {
  if (is_pow2(in_size)) {
    for (ptrdiff_t i = 0; i < out_size; i++) {
      out[i] = ConvertSat<T>(in[i & (in_size - 1)]);
    }
  } else {
    ptrdiff_t i = 0;
    for (; i + in_size <= out_size; i += in_size) {
      for (ptrdiff_t j = 0; j < in_size; j++) {
        out[i + j] = ConvertSat<T>(in[j]);
      }
    }
    for (ptrdiff_t j = 0; i < out_size; i++, j++) {
      out[i] = ConvertSat<T>(in[j]);
    }
  }
}

template <typename T, typename U>
void Broadcast(
      int ndim, T *out,
      const int64_t *out_shape,
      const int64_t *out_strides,
      const U *in,
      const int64_t *in_strides) {
  if (ndim == 0) {
    *out = ConvertSat<T>(*in);
  } else if (ndim == 1) {
    int64_t in_stride = *in_strides;
    int64_t out_stride = *out_strides;
    for (ptrdiff_t i = 0, extent = out_shape[0]; i < extent; i++) {
      out[i * out_stride] = ConvertSat<T>(in[i * in_stride]);
    }
  } else {
    int64_t in_stride = *in_strides;
    int64_t out_stride = *out_strides;
    for (ptrdiff_t i = 0, extent = out_shape[0]; i < extent; i++) {
      Broadcast(
        ndim - 1,
        out + i * out_stride,
        out_shape + 1,
        out_strides + 1,
        in + i * in_stride,
        in_strides + 1);
    }
  }
}

template <>
void ConstantValue<CPUBackend>::RunImpl(Workspace &ws) {
  auto &output = ws.Output<CPUBackend>(0);
  const auto& out_shapes = output.shape();
  int nsamples = out_shapes.size();
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
            auto out_shape = out_shapes.tensor_shape_span(s);
            auto in_shape = fill_value.tensor_shape_span(s);
            if (in_shape.back() == volume(in_shape)) {
              RepeatInner(out_data, volume(out_shape), fill_value_data, volume(in_shape));
            } else {
              TensorShape<> in_strides;
              in_strides.resize(out_shape.size());
              int i = in_shape.size() - 1;
              int j = out_shape.size() - 1;
              ptrdiff_t stride = 1;
              for (; i >= 0 && j >= 0; i--, j--) {
                if (in_shape[i] == out_shape[j]) {
                  in_strides[j] = stride;
                  stride *= in_shape[i];
                } else {
                  assert(in_shape[i] == 1);
                  in_strides[j] = 0;
                }
              }
              for (; j >= 0; j--) {
                in_strides[j] = 0;
              }
              int ndim = out_shape.size();
              TensorShape<> out_strides;
              kernels::CalcStrides(out_strides, out_shape);
              Broadcast(
                  ndim,
                  out_data, out_shape.data(), out_strides.data(),
                  fill_value_data, in_strides.data());
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
          auto out_sz = out_shapes.tensor_size(s);
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
