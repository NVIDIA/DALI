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

#ifndef DALI_OPERATORS_GENERIC_ONE_HOT_H_
#define DALI_OPERATORS_GENERIC_ONE_HOT_H_

#include <vector>
#include <string>

#include "dali/pipeline/operator/operator.h"
#include "dali/kernels/kernel_params.h"
#include "dali/core/tensor_view.h"
#include "dali/core/static_switch.h"

#define ONE_HOT_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float, double)  // NOLINT

namespace dali {

namespace detail {

template<typename Out, typename In>
void DoOneHot(kernels::OutTensorCPU<Out, DynamicDimensions> output,
              kernels::InTensorCPU<In, DynamicDimensions> input, int num_classes,
              same_as_t<Out> on_value, same_as_t<Out> off_value, int axis) {
  auto in = input.data;
  auto out = output.data;
  auto volume_outer = volume(output.shape.begin(), output.shape.begin() + axis);
  auto volume_inner = volume(output.shape.begin() + axis + 1, output.shape.end());
  for (int i = 0; i < volume(output.shape); ++i) {
    out[i] = off_value;
  }
  for (int64_t outer_coord = 0; outer_coord < volume_outer; outer_coord++) {
    for (int64_t inner_coord = 0; inner_coord < volume_inner; inner_coord++) {
      int cls = in[outer_coord * volume_inner + inner_coord];
      if (cls < 0 || cls >= num_classes)
        continue;
      out[outer_coord * volume_inner * num_classes + cls * volume_inner + inner_coord] = on_value;
    }
  }
}
}  // namespace detail

class OneHot : public Operator<CPUBackend> {
 public:
  inline explicit OneHot(const OpSpec &spec)
      : Operator<CPUBackend>(spec), num_classes_(spec.GetArgument<int64_t>("num_classes")),
        axis_(spec.GetArgument<int>("axis")),
        output_type_(spec.GetArgument<DALIDataType>(arg_names::kDtype)),
        on_value_(spec.GetArgument<float>("on_value")),
        off_value_(spec.GetArgument<float>("off_value")) {
    if (spec.HasArgument("axis_name")) {
      auto axis_name = spec.GetArgument<std::string>("axis_name");
      DALI_ENFORCE(axis_name.length() == 1,
                   make_string("Unsupported axis_name value. It must be a single "
                               "character, got \"", axis_name, "\" instead"));
      new_axis_name_ = axis_name[0];
    }
  }

  inline ~OneHot() override = default;

  DISABLE_COPY_MOVE_ASSIGN(OneHot);

  USE_OPERATOR_MEMBERS();
  using Operator<CPUBackend>::RunImpl;

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const HostWorkspace &ws) override;
  void RunImpl(HostWorkspace &ws) override;

  TensorLayout GetOutputLayout(const HostWorkspace &ws, int placement_axis,
                               int output_sample_dim) const;

  int num_classes_;
  int axis_;
  const DALIDataType output_type_;
  float on_value_;
  float off_value_;
  char new_axis_name_ = 0;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_ONE_HOT_H_
