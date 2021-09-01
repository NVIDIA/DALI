// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/generic/lookup_table.h"


namespace dali {

namespace {

template <typename Output, typename Input>
void LookupValuesImpl(ThreadPool &tp, TensorVector<CPUBackend> &output,
                      const TensorVector<CPUBackend> &input, const TensorListShape<> &shape,
                      const Output *lookup_table, const Output default_value) {
  for (int sample_idx = 0; sample_idx < shape.num_samples(); sample_idx++) {
    auto data_size = shape.tensor_size(sample_idx);
    auto *out_data = output[sample_idx].mutable_data<Output>();
    const auto *in_data = input[sample_idx].data<Input>();
    tp.AddWork(
        [=](int thread_id) {
          for (int64_t i = 0; i < data_size; i++) {
            DoLookup<CPUBackend>(out_data[i], in_data[i], lookup_table, default_value);
          }
        },
        data_size);
  }
}

}  // namespace

template <>
void LookupTable<CPUBackend>::RunImpl(HostWorkspace &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  const auto &shape = input.shape();
  auto &output = ws.OutputRef<CPUBackend>(0);
  output.SetLayout(input.GetLayout());

  auto &tp = ws.GetThreadPool();

  TYPE_SWITCH(input.type().id(), dali::type2id, Input, LUT_IN_TYPES, (
    TYPE_SWITCH(output_type_, dali::type2id, Output, LUT_OUT_TYPES, (

      auto default_value = ConvertSat<Output>(default_value_f_);
      const auto *lookup_table = static_cast<Output *>(value_mem_.get());

      LookupValuesImpl<Output, Input>(tp, output, input, shape, lookup_table, default_value);

    ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)); );       // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())); );     // NOLINT

  tp.RunAll();
}

DALI_REGISTER_OPERATOR(LookupTable, LookupTable<CPUBackend>, CPU);

DALI_SCHEMA(LookupTable)
  .DocStr(R"code(Maps the input to output by using a lookup table that is specified by
``keys`` and ``values``, and a ``default_value`` for unspecified keys.

For example when ``keys`` and ``values`` are used to define the lookup table in the following way::

  keys[] =   {0,     2,   3,   4,   5,    3}
  values[] = {0.2, 0.4, 0.5, 0.6, 0.7, 0.10}
  default_value = 0.99

  0 <= i < max(keys)
  lut[i] = values[keys.index[i]]   if i in keys
  lut[i] = default_value           otherwise

the operator creates the following table::

  lut[] = {0.2, 0.99, 0.4, 0.10, 0.6, 0.7}  // only last occurrence of a key is considered

and produces the output according to this formula::

  Output[i] = lut[Input[i]]   if 0 <= Input[i] <= len(lut)
  Output[i] = default_value   otherwise

Here is a practical example, considering the table defined above::

  Input[] =  {1,      4,    1,   0,  100,   2,     3,   4}
  Output[] = {0.99, 0.6, 0.99, 0.2, 0.99, 0.4,  0.10, 0.6}

.. note::
  Only integer types can be used as inputs for this operator.
)code")
  .NumInput(1)
  .NumOutput(1)
  .AllowSequences()
  .SupportVolumetric()
  .AddOptionalArg("dtype",
    R"code(Output data type.)code",
    DALI_FLOAT)
  .DeprecateArgInFavorOf("output_dtype", "dtype")  // deprecated since 0.24dev
  .AddOptionalArg("default_value",
    R"code(Default output value for keys that are not present in the table.)code",
    0.0f)
  .AddOptionalArg("keys",
    R"code(A list of input values (keys) in the lookup table.

The length of ``keys`` and ``values`` argument must match. The values in ``keys`` should be in the
[0, )code" + std::to_string(LookupTable<CPUBackend>::kMaxKey) + " ] range.",
    std::vector<int>{})
  .AddOptionalArg("values",
    R"code(A list of mapped output ``values`` for each ``keys`` entry.

The length of the ``keys`` and the ``values`` argument must match.
)code",
    std::vector<float>{});

}  // namespace dali
