// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#ifndef DALI_OPERATORS_SEQUENCE_PER_FRAME_H_
#define DALI_OPERATORS_SEQUENCE_PER_FRAME_H_

#include <vector>
#include "dali/core/common.h"
#include "dali/core/tensor_layout.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class PerFrame : public StatelessOperator<Backend> {
 public:
  inline explicit PerFrame(const OpSpec &spec)
      : StatelessOperator<Backend>(spec), replace_(spec.GetArgument<bool>("replace")) {}

 protected:
  bool CanInferOutputs() const override {
    // Return false to prevent executor from allocating memory for the output,
    // even though the output shape could be inferred, as it is same as input
    return false;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    const auto &input = ws.Input<Backend>(0);
    output_desc.resize(1);
    output_desc[0].type = input.type_info().id();
    output_desc[0].shape = input.shape();
    return false;
  }

  void RunImpl(Workspace &ws) override {
    auto &in = ws.Input<Backend>(0);
    auto &out = ws.Output<Backend>(0);
    out.ShareData(in);
    auto layout = in.GetLayout();
    auto sample_dim = in.sample_dim();
    out.SetLayout(GetOutputLayout(layout, sample_dim));
  }

  TensorLayout GetOutputLayout(const TensorLayout &layout, int sample_dim) const {
    if (layout.empty()) {
      DALI_ENFORCE(sample_dim > 0, "Cannot mark zero-dimensional input as a sequence.");
      TensorLayout ret;
      ret.resize(sample_dim, '*');
      ret[0] = 'F';
      return ret;
    }
    if (replace_) {
      TensorLayout ret(layout);
      ret[0] = 'F';
      return ret;
    }
    DALI_ENFORCE(layout[0] == 'F', make_string("Per-frame argument input must be a sequence. The "
                                               "input layout should start with 'F', got: ",
                                               layout));
    return layout;
  }

  USE_OPERATOR_MEMBERS();

 private:
  bool replace_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_SEQUENCE_PER_FRAME_H_
