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

template <typename Backend>
class OneHot : public Operator<Backend> {
 public:
  explicit OneHot(const OpSpec &spec)
      : Operator<Backend>(spec), num_classes_(spec.GetArgument<int64_t>("num_classes")),
        axis_(spec.GetArgument<int>("axis")),
        output_type_(spec.GetArgument<DALIDataType>(arg_names::kDtype)),
        on_value_(spec.GetArgument<float>("on_value")),
        off_value_(spec.GetArgument<float>("off_value")) {
    if (spec.HasArgument("axis_name")) {
      auto axis_name = spec.GetArgument<std::string>("axis_name");
      DALI_ENFORCE(axis_name.length() == 1,
                   make_string("Unsupported axis_name value. It must be a single "
                               "character, got \"", axis_name, "\" instead."));
      new_axis_name_ = axis_name[0];
    }
  }

  DISABLE_COPY_MOVE_ASSIGN(OneHot);

  USE_OPERATOR_MEMBERS();

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    const auto &input = ws.template InputRef<Backend>(0);
    int input_sample_dim = input.shape().sample_dim();
    int num_samples = input.shape().num_samples();
    DALI_ENFORCE(-1 <= axis_ && axis_ <= input_sample_dim,
                 make_string("Provided axis is outside of allowed range, got: ", axis_,
                             ", expected to be in range: [-1, ", input_sample_dim, "]."));

    // Legacy scalar-like support only if the `axis` parameter was not provided
    bool all_scalars = !this->spec_.ArgumentDefined("axis");
    for (int i = 0; all_scalars && i < num_samples; i++) {
      all_scalars = all_scalars && is_scalar(input.shape()[i]);
    }

    int output_sample_dim = all_scalars ? 1 : input_sample_dim + 1;
    output_desc.resize(1);

    output_desc[0].shape.resize(num_samples, output_sample_dim);

    const auto& shape = input.shape();
    for (int i = 0; i < num_samples; i++) {
      output_desc[0].shape.set_tensor_shape(i, determine_shape(shape[i], output_sample_dim));
    }
    TYPE_SWITCH(output_type_, type2id, DType, ONE_HOT_TYPES, ({
                  TypeInfo type;
                  type.SetType<DType>(output_type_);
                  output_desc[0].type = type;
                }),
                DALI_FAIL(make_string("Unsupported output type: ", output_type_))) // NOLINT
    return true;
  };

  TensorLayout GetOutputLayout(const workspace_t<Backend> &ws, int placement_axis,
                               int output_sample_dim) {
    if (!new_axis_name_) {
      return {};
    }
    const auto &input = ws.template InputRef<Backend>(0);
    auto in_layout = input.GetLayout();
    // .size method returns uint_8 which doesn't work well when 0 is printed in error message
    int in_layout_size = in_layout.size();
    // Handles the legacy 'multi-dimensional' scalars-like
    if (output_sample_dim == 1) {
      return TensorLayout(&new_axis_name_, 1);
    }
    if (in_layout_size + 1 == output_sample_dim) {
      return in_layout.first(placement_axis) + TensorLayout(&new_axis_name_, 1) +
             in_layout.last(in_layout_size - placement_axis);
    }
    DALI_FAIL(make_string("Input layout mismatch - expected input layout to be of size ",
                          output_sample_dim - 1, " but instead got \"", in_layout,
                          "\", which is of size ", in_layout_size, "."));
  }

  inline int get_placement_axis(int output_sample_dim) {
    return axis_ < 0 ? output_sample_dim - 1 : axis_;
  }

  template <int ndims>
  bool is_scalar(const TensorShape<ndims> &shape) {
    return volume(shape) == 1;
  }

  TensorShape<> determine_shape(const TensorShape<> &in_shape, int output_sample_dim) {
    if (output_sample_dim == 1) {
      return {num_classes_};
    }
    auto placement_axis = get_placement_axis(output_sample_dim);
    int outer_axes = placement_axis;
    int inner_axes = in_shape.sample_dim() - placement_axis;
    auto outer = in_shape.first(outer_axes);
    auto inner = in_shape.last(inner_axes);
    return shape_cat(shape_cat(outer, num_classes_), inner);
  }

  int num_classes_;
  int axis_;
  const DALIDataType output_type_;
  float on_value_;
  float off_value_;
  char new_axis_name_ = 0;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_ONE_HOT_H_
