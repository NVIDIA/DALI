// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_MATH_NORMALIZE_NORMALIZE_H_
#define DALI_OPERATORS_MATH_NORMALIZE_NORMALIZE_H_

#include <memory>
#include <sstream>
#include <vector>

#include "dali/core/any.h"
#include "dali/core/tensor_shape.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/operators/util/diag_msg.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

#define DALI_NORMALIZE_INPUT_TYPES (int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, float)
#define DALI_NORMALIZE_OUTPUT_TYPES DALI_NORMALIZE_INPUT_TYPES

template <typename Backend>
class Normalize;

template <typename Backend>
class NormalizeBase : public Operator<Backend> {
 public:
  explicit NormalizeBase(const OpSpec &spec) : Operator<Backend>(spec) {
    has_tensor_mean_ = spec.HasTensorArgument("mean");
    has_scalar_mean_ = spec.HasArgument("mean") && !has_tensor_mean_;
    has_tensor_stddev_ = spec.HasTensorArgument("stddev");
    has_scalar_stddev_ = spec.HasArgument("stddev") && !has_tensor_stddev_;

    batch_norm_ = spec.GetArgument<bool>("batch");
    has_axes_arg_ = spec.HasArgument("axes");
    has_axis_names_arg_ = spec.HasArgument("axis_names");
    shift_ = spec.GetArgument<float>("shift");
    scale_ = spec.GetArgument<float>("scale");
    epsilon_ = spec.GetArgument<float>("epsilon");
    degrees_of_freedom_ = spec.GetArgument<int>("ddof");
    output_type_ = spec.GetArgument<DALIDataType>("dtype");

    DALI_ENFORCE(!has_axes_arg_ || !has_axis_names_arg_,
      "Normalize: Arguments `axes` and `axis_names` are mutually exclusive");

    if (has_scalar_mean_ && has_scalar_stddev_) {
      DALI_ENFORCE(!has_axes_arg_ && !has_axis_names_arg_,
        "Normalize: Axes must not be specified when both mean and standard deviation are scalars");
    }

    if (has_tensor_mean_ || has_tensor_stddev_) {
      DALI_ENFORCE(!batch_norm_, "Normalize: Batch normalization cannot be used with parameters "
      "specified as TensorList inputs");
    }
    mean_.set_pinned(false);
    inv_stddev_.set_pinned(false);
  }

  Normalize<Backend> &This() noexcept
  { return static_cast<Normalize<Backend>&>(*this); }

  const Normalize<Backend> &This() const noexcept
  { return static_cast<const Normalize<Backend>&>(*this); }

  bool CanInferOutputs() const override { return true; }

  bool SetupImpl(std::vector<OutputDesc> &output_descs, const workspace_t<Backend> &ws) override {
    const auto &input = ws.template InputRef<Backend>(0);
    data_shape_ = input.shape();
    output_descs.resize(1);
    output_descs[0] = { data_shape_, TypeTable::GetTypeInfo(output_type_) };
    input_type_ = input.type().id();

    SetupAxes(ws);
    TYPE_SWITCH(input_type_, type2id, InputType, DALI_NORMALIZE_INPUT_TYPES, (
      TYPE_SWITCH(output_type_, type2id, OutputType, DALI_NORMALIZE_OUTPUT_TYPES, (
        This().template SetupTyped<OutputType, InputType>(ws);
      ), (DALI_FAIL(make_string("Normalize: unsupported output type: ", output_type_))))  // NOLINT
    ), (DALI_FAIL(make_string("Normalize: unsupported input type: ", input_type_))));    // NOLINT
    return true;
  }

  void RunImpl(workspace_t<Backend> &ws) override {
    TYPE_SWITCH(input_type_, type2id, InputType, DALI_NORMALIZE_INPUT_TYPES, (
      TYPE_SWITCH(output_type_, type2id, OutputType, DALI_NORMALIZE_OUTPUT_TYPES, (
        This().template RunTyped<OutputType, InputType>(ws);
      ), (DALI_FAIL("Normalize: ureachable code - Run without matching Setup?")))  // NOLINT
    ), (DALI_FAIL("Normalize: ureachable code - Run without matching Setup?")));   // NOLINT
  }


  void UseAllAxes() {
    int dim = data_shape_.sample_dim();
    axes_.resize(dim);
    std::iota(axes_.begin(), axes_.end(), 0);
    axis_mask_ = (1_u64 << dim) - 1;
    GetParamShapeFromAxes();
  }

  void SetupAxes(const workspace_t<Backend> &ws) {
    int dim = data_shape_.sample_dim();
    if (has_scalar_mean_ && has_scalar_stddev_) {
      UseAllAxes();
      return;
    }

    const OpSpec &spec = this->spec_;

    ConsumeArguments(ws);

    if (has_axes_arg_) {
      axes_ = spec.GetRepeatedArgument<int>("axes");
      DALI_ENFORCE(!axes_.empty(),
        "Normalize `axes` argument must specify at least one reduction axis.");
      for (auto axis : axes_) {
        DALI_ENFORCE(axis >= 0 && axis < dim, make_string("Normalize: axis index ", axis,
        " is out of valid range 0..", dim-1));
      }
      SetAxisMask();
      if (!has_tensor_mean_ && !has_tensor_stddev_)
        GetParamShapeFromAxes();
    } else if (has_axis_names_arg_) {
      TensorLayout names = spec.GetArgument<TensorLayout>("axis_names");
      const auto &input = ws.template InputRef<Backend>(0);
      auto dim_idx = GetDimIndices(input.GetLayout(), names);
      axes_ = dim_idx.to_vector();
      SetAxisMask();
      if (!has_tensor_mean_ && !has_tensor_stddev_)
        GetParamShapeFromAxes();
    } else if (has_tensor_mean_ || has_tensor_stddev_) {
      GetAxesFromParamShape();
    } else {
      UseAllAxes();
      return;
    }
    CheckParamShape();
  }


  void GetParamShapeFromAxes() {
    int dim = data_shape_.sample_dim();
    int n = data_shape_.num_samples();
    if (batch_norm_) {
      param_shape_.resize(1, dim);
      DALI_ENFORCE(data_shape_.num_samples() > 0, "Normalize: Got an empty batch!");
      param_shape_.set_tensor_shape(0, data_shape_[0]);
      for (int axis = 0; axis < dim; axis++) {
        if (IsReducedAxis(axis))
          param_shape_.tensor_shape_span(0)[axis] = 1;
      }
      for (int i = 1; i < n; i++) {
        for (int axis = 0; axis < dim; axis++) {
          if (!IsReducedAxis(axis))
            DALI_ENFORCE(data_shape_[i][axis] == data_shape_[0][axis], make_string(
              "Normalize: Batch normalization requires that non-reduced dimensions have equal "
              "extent in all samples in the batch. Got sample #", i, " with shape ",
              data_shape_[i], " which has a different extent  (", data_shape_[i][axis],
              ") in axis ", axis, " than sample #0, which was "
              "of shape ", data_shape_[0], " (", data_shape_[0][axis], " in axis ", axis, ")"));
        }
      }
    } else {
      param_shape_.resize(n, dim);
      for (int i = 0; i < n; i++) {
        param_shape_.set_tensor_shape(i, data_shape_[i]);
        for (int axis = 0; axis < dim; axis++) {
          if (IsReducedAxis(axis))
            param_shape_.tensor_shape_span(i)[axis] = 1;
        }
      }
    }
  }

  void ConsumeArguments(const workspace_t<Backend> &ws) {
    if (has_tensor_mean_) {
      mean_input_ = view<const float>(ws.ArgumentInput("mean"));
      param_shape_ = mean_input_.shape;
    }

    if (has_tensor_stddev_) {
      stddev_input_ = view<const float>(ws.ArgumentInput("stddev"));
      if (!has_tensor_mean_)  // if not taken from `mean` - use `stddev` shape
        param_shape_ = stddev_input_.shape;
    }

    if (has_tensor_mean_ && has_tensor_stddev_) {
      if (mean_input_.shape != stddev_input_.shape) {
        auto msg = ShapeMismatchMsg(mean_input_.shape, stddev_input_.shape);
        DALI_FAIL(make_string("Normalize: When providing both `mean` and `stddev`, "
          "their shapes must match.\n", msg));
      }
    }
  }

  void GetAxesFromParamShape() {
    axes_.clear();
    for (int d = 0; d < param_shape_.sample_dim(); d++) {
      if (is_degenerate_dim(param_shape_, d))
          axes_.push_back(d);
    }
    axis_mask_ = to_bit_mask(axes_);
  }

  void CheckParamShape() {
    int dim = param_shape_.sample_dim();
    int n = data_shape_.num_samples();
    TensorShape<> expected;
    expected.resize(dim);
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dim; d++) {
        expected[d] = IsReducedAxis(d) ? 1 : data_shape_.tensor_shape_span(i)[d];
      }
      int param_idx = batch_norm_ ? 0 : i;
      DALI_ENFORCE(param_shape_[param_idx] == expected, make_string(
        "Normalize: At sample ", i, ": parameter shape: ", param_shape_[param_idx],
        " does not match the reduced input sample shape which is : ", expected));
    }
  }

  void SetAxisMask() {
    axis_mask_ = 0;
    for (auto axis : axes_)
      axis_mask_ |= 1_u64 << axis;
  }

  bool IsReducedAxis(int axis) const noexcept {
    return axis_mask_ & (1_u64 << axis);
  }

  bool ShouldCalcMean() const noexcept { return !has_tensor_mean_ && !has_scalar_mean_; }
  bool ShouldCalcStdDev() const noexcept { return !has_tensor_stddev_ && !has_scalar_stddev_; }
  bool IsFullReduction() const noexcept {
    int ndim = data_shape_.sample_dim();
    return axis_mask_== ((1_u64 << ndim) - 1);
  }

 protected:
  TensorListShape<> data_shape_;
  TensorListShape<> param_shape_;

  // NOTE: StorageCPU, because these are argument inputs - this can be reworked
  // when we have either GPU argument inputs or named regular inputs
  TensorListView<StorageCPU, const float> mean_input_, stddev_input_;


  TensorList<Backend> mean_, inv_stddev_;
  bool has_tensor_mean_, has_tensor_stddev_ = false;
  bool has_scalar_mean_, has_scalar_stddev_ = false;
  bool batch_norm_ = false;
  bool has_axes_arg_ = false;
  bool has_axis_names_arg_ = false;
  float shift_ = 0;
  float scale_ = 1;
  float epsilon_ = 0;  //!< Added to variance for regularization
  int degrees_of_freedom_ = 0;  //!< For Bessel's correction
  DALIDataType input_type_ = DALI_NO_TYPE, output_type_ = DALI_FLOAT;
  std::vector<int> axes_;
  uint64_t axis_mask_ = 0;
};

}  // namespace dali

#endif  // DALI_OPERATORS_MATH_NORMALIZE_NORMALIZE_H_
