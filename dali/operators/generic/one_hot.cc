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


DALI_SCHEMA(OneHot)
  .DocStr(R"code(Produces a one-hot encoding of the input.

Adds a new axis or converts scalar input into an axis of ``num_classes`` elements.

For given input coordinate ``(x0, x1, ..., xn)``, and ``axis = k``, the output sample is specified as::

  cls = input[x0, x1, ..., xn]
  output[x0, x1, ..., xk-1, i, xk, ..., xn] = on_value if i == cls else off_value

for all ``i`` in range ``[0, num_classes)``.

For scalars, the output is set to ``on_value`` at the index taken from ``input`` and
``off_value`` elsewhere::

  output[i] = on_value if i == input else off_value

For backward compatibility, any input in which all tensors have only one element
(regardless of the number of dimensions) is considered scalar. Legacy interpretation of tensors
as scalars is not supported if ``axis`` argument is specified.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("num_classes", R"code(Number of all classes in the data.)code", 0)
  .AddOptionalArg<int>("axis", R"code(Dimension to place the one-hot encoding axis of `num_classes`
size. By default it's appended as the last dimension for non-scalar inputs. For scalar inputs,
it becomes the only dimension.)code", -1)
  .AddOptionalArg(arg_names::kDtype, R"code(Output data type.)code", DALI_FLOAT)
  .AddOptionalArg("on_value",
                  R"code(Value that will be used to fill the output to indicate given class
in the corresponding input coordinate.

This value will be cast to the ``dtype`` type.)code", 1.f)
  .AddOptionalArg("off_value",
                  R"code(Value that will be used to fill the output to indicate the lack of given
class in the corresponding input coordinate.

This value will be cast to the ``dtype`` type.)code", 0.f)
  .AddOptionalArg<std::string>("axis_name",
                  R"code(Single character that will be used as a name for the newly added
dimension in the output layout. If no character is provided, the output layout will be
empty.)code", nullptr);

class OneHotCPU : public OneHot<CPUBackend> {
 public:
  explicit OneHotCPU(const OpSpec &spec)
      : OneHot<CPUBackend>(spec) {}

  ~OneHotCPU() override = default;

  void RunImpl(workspace_t<CPUBackend> &ws) override;

  USE_OPERATOR_MEMBERS();
};

void OneHotCPU::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  auto &output = ws.template OutputRef<CPUBackend>(0);
  auto &tp = ws.GetThreadPool();
  auto in_shape = input.shape();
  auto num_samples = in_shape.num_samples();
  int output_sample_dim = output.shape().sample_dim();
  int placement_axis = get_placement_axis(output_sample_dim);
  output.SetLayout(GetOutputLayout(ws, placement_axis, output_sample_dim));
  TYPE_SWITCH(input.type().id(), type2id, InputType, ONE_HOT_TYPES, (
    TYPE_SWITCH(output_type_, type2id, OutputType, ONE_HOT_TYPES, (

    auto in_tensor = view<const InputType, DynamicDimensions>(input);
    auto out_tensor = view<OutputType, DynamicDimensions>(output);
    for (int sample_id = 0; sample_id < num_samples; ++sample_id) {
      tp.AddWork(
              [&, sample_id](int thread_id) {
                  auto in = in_tensor[sample_id];
                  auto out = out_tensor[sample_id];
                  detail::DoOneHot(out, in, num_classes_, on_value_, off_value_, placement_axis);
              }, in_shape.tensor_size(sample_id));
    }
    tp.RunAll();
  ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())))  // NOLINT
}

DALI_REGISTER_OPERATOR(OneHot, OneHotCPU, CPU);

}  // namespace dali
