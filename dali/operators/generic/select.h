// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_GENERIC_SELECT_H_
#define DALI_OPERATORS_GENERIC_SELECT_H_

#include <vector>
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/kernels/common/scatter_gather.h"

namespace dali {

template <typename Backend>
class Select : public Operator<Backend> {
 public:
  explicit Select(const OpSpec &spec) : Operator<Backend>(spec), input_idx_("input_idx", spec) {
    has_layout_arg_ = spec.TryGetArgument(layout_, "layout");
  }

  bool CanInferOutputs() const override { return true; }

  USE_OPERATOR_MEMBERS();

  bool SetupImpl(vector<OutputDesc> &outputs, const workspace_t<Backend> &ws) override {
    auto &inp0 = ws.template InputRef<Backend>(0);
    int num_inp = spec_.NumRegularInput();
    int sample_dim = inp0.sample_dim();
    for (int i = 1; i < spec_.NumRegularInput(); i++) {
      auto &inp = ws.template InputRef<Backend>(i);
      DALI_ENFORCE(inp.type() == inp0.type(), make_string(
        "All inputs must have the same type. "
        "Got: ", inp0.type().id(), " and ", inp.type().id()));

      DALI_ENFORCE(inp.sample_dim() == sample_dim, make_string(
        "All inputs must have the same number of dimensions. "
        "Got: ", sample_dim, " and ", inp.sample_dim()));
    }

    if (has_layout_arg_) {
      DALI_ENFORCE(layout_.size() == sample_dim, make_string("The layout '", layout_, "' is not "
        "a valid layout for ", sample_dim, "-D tensors."));
    }

    int num_samples = inp0.ntensor();

    outputs.resize(1);
    outputs[0].shape.resize(num_samples, sample_dim);
    outputs[0].type = inp0.type();

    input_idx_.Acquire(spec_, ws, num_samples);

    TensorShape<> empty_shape;
    empty_shape.resize(sample_dim);

    for (int i = 0; i < num_samples; i++) {
      int idx = *input_idx_[i].data;
      bool is_valid_idx = (sample_dim > 0 && idx < 0) || (idx >= 0 && idx < num_inp);
      DALI_ENFORCE(is_valid_idx, make_string("Invalid input index for sample ", i, ": ", idx));
      if (idx < 0) {
        outputs[0].shape.set_tensor_shape(i, empty_shape);
      } else {
        auto &inp = ws.template InputRef<Backend>(idx);
        outputs[0].shape.set_tensor_shape(i, inp.tensor_shape(i));
      }
    }
    return true;
  }

  void RunImpl(workspace_t<Backend> &ws) override;

 private:
  void SetOutputLayout(workspace_t<Backend> &ws) {
    if (has_layout_arg_) {
      ws.template OutputRef<Backend>(0).SetLayout(layout_);
    } else {
      for (int i = 0; i < spec_.NumRegularInput(); i++) {
        auto &inp = ws.template InputRef<Backend>(i);
        auto layout = inp.GetLayout();
        if (!layout.empty()) {
            ws.template OutputRef<Backend>(0).SetLayout(layout);
            break;
        }
      }
    }
  }

  ArgValue<int> input_idx_;
  TensorLayout layout_;
  bool has_layout_arg_;
  kernels::ScatterGatherGPU sg_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_SELECT_H_
