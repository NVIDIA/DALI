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
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_REGISTER_OPERATOR(OneHot, OneHot, CPU);


DALI_SCHEMA(OneHot)
  .DocStr(R"code(Produces a one-hot encoding of the input.

If the input is not a scalar (tensor consisting from one value per sample), the operator
will fail.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("num_classes", R"code(Number of all classes in the data.)code", 0)
  .AddOptionalArg(arg_names::kDtype, R"code(Output data type.)code", DALI_FLOAT)
  .AddOptionalArg("on_value",
                  R"code(Value that will be used to fill the output when ``input[j] = i``.

This value will be cast to the ``dtype`` type.)code", 1.f)
  .AddOptionalArg("off_value",
                  R"code(Value that will be used to fill the output when ``input[j] != i``.

This value will be cast to the ``dtype`` type.)code", 0.f);

namespace {

template<int ndims>
bool is_scalar(TensorShape<ndims> shape) {
  return volume(shape) == 1;
}


TensorShape<DynamicDimensions>
determine_shape(TensorShape<DynamicDimensions> in_shape, int nclasses) {
  if (is_scalar(in_shape)) {
    return {nclasses};
  }
  return shape_cat(in_shape, nclasses);
}

}  // namespace

bool OneHot::SetupImpl(std::vector<OutputDesc> &output_desc, const HostWorkspace &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  output_desc.resize(1);
  output_desc[0].shape = uniform_list_shape(batch_size_,
                                            determine_shape(input[0].shape(), num_classes_));
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
  auto in_shape = input.shape();
  TYPE_SWITCH(input.type().id(), type2id, InputType, ONE_HOT_TYPES, (
          TYPE_SWITCH(output_type_, type2id, OutputType, ONE_HOT_TYPES, (

          auto in_tensor = view<const InputType, DynamicDimensions>(input);
          auto out_tensor = view<OutputType, DynamicDimensions>(output);
          for (int sample_id = 0; sample_id < batch_size_; ++sample_id) {
            tp.AddWork(
                    [&, sample_id](int thread_id) {
                        auto in = in_tensor[sample_id];
                        auto out = out_tensor[sample_id];
                        detail::DoOneHot(out, in, num_classes_, on_value_, off_value_,
                                         0 ? is_scalar(in.shape) : in.shape.sample_dim(), 0);
                    }, in_shape.tensor_size(sample_id));
          }
          tp.RunAll();
  ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())))  // NOLINT
}

}  // namespace dali
