// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include <vector>
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel_manager.h"

namespace dali {

template <typename Backend>
class Normalize : public Operator<Backend> {
 public:
  explicit Normalize(const OpSpec &spec) : Operator<Backend>(spec) {
    has_tensor_mean_ = spec.HasTensorArgument("mean");
    has_scalar_mean_ = spec.GetArgument("mean") && !has_tensor_mean_;

    calc_mean_ = !spec.HasArgument("mean") && !spec.HasTensorArgument("mean");
    calc_stddev_ = !spec.HasArgument("stddev") && !spec.HasTensorArgument("stddev");
    batch_norm_ = spec.GetArgument<bool>("batch");
    has_axes_arg_ = spec.HasTensorArgument("axes");
    has_axis_names_arg_ = spec.HasTensorArgument("axis_names");
    shift_ = spec.GetArgument<float>("shift");
    scale_ = spec.GetArgument<float>("scale");

    DALI_ENFORCE(!has_axes_arg_ || !has_axis_names_arg_,
      "Normalize: Arguments `axes` and `axis_names` are mutually exclusive");

  }

  void SetupImpl(std::vector<OutputDesc> &output_descs, const workspace_t<Backend> &ws) {
    input_shape_ = ws.InputRef(0).shape();
    output_descs.resize(1);
    output_descs[0] = { input_shape_, TypeTable::GetTypeInfo(output_type_) };
  }

  void SetupAxes(const workspace_t<Backend> &ws) {
    int dim = input_shape_.sample_dim();

    if (has_axes_arg_) {
      axes_ = spec_.GetRepeatedArgument<int>("axes");
      for (auto axis : axes_) {
        DALI_ENFORCE(axis >= 0 && axis < dim, make_string("Normalize: axis index ", axis,
        " is out of valid range 0..", dim-1));
      }
      SetAxisMask();
      GetParamShapesFromAxes();
    } else if (has_axis_names_arg_) {
      axes_ = GetDimIndices(InputLayout(ws, 0), spec.GetArgument<TensorLayout>("axis_names"));
      SetAxisMask();
      GetParamShapesFromAxes();
    } else {
      GetParamShapesFromArgs();
      GetAxesFromParamShapes(ws);
    }
    CheckParamShape();
  }


  void GetParamShapesFromAxes() {
    for (int i = 0; i < input_shape_.num_samples(); i++) {
      param_shape_.set_tensor_shape(i, input_shape_[i])
      for (int axis = 0; axis < param_shape_.dim; axis++) {
        if (IsReducedAxis(axis))
          param_shape_[axis] = 1;
      }
    }
  }

  void GetAaxesFromParamShapes(const workspace_t<Backend> &ws) {
    if (spec_.HasTensorArgument("mean")) {
      mean_input_ = view<float>(ws.ArgumentInput("mean"));
    }
    if (spec_.HasTensorArgument("stddev")) {
      stddev_input_ = view<float>(ws.ArgumentInput("stddev"));
    }

    if (!mean_input_.empty() && !stddev_input_.empty())
      DALI_ENFORCE("When providing both `mean` and `stddev`, their shapes must match");
  }

  void CheckParamShape() {

  }

  void SetAxisMask() {
    axis_mask_ = 0;
    for (auto axis : axes_)
      axis_mask_ |= 1 << axis;
  }

  bool IsReducedAxis(int axis) const noexcept {
    return axis_mask_ & (1 << axis);
  }

  kernels::KernelManager kmgr_;

  bool calc_mean_ = true, calc_stddev_ = true;
  TensorListShape<> input_shape_;
  TensorListShape<> param_shape_;
  TensorListView<StorageCPU, float> mean_input_, stddev_input_;
  bool has_tensor_mean_, has_tensor_stddev_ = false;
  bool batch_norm_ = false;
  bool has_axes_arg_ = false;
  bool has_axis_names_arg_ = false;
  DALIDataType output_type_ = DALI_FLOAT;
  float shift_ = 0;
  float scale_ = 1;
  std::vector<int> axes_;
  int axis_mask_ = 0;
};

}  // namespace dali

#endif  // DALI_OPERATORS_MATH_NORMALIZE_NORMALIZE_H_
