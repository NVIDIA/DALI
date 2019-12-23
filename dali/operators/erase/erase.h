// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_ERASE_ERASE_H_
#define DALI_OPERATORS_ERASE_ERASE_H_

#include <tuple>
#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/tensor_shape.h"
#include "dali/kernels/scratch.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class Erase : public Operator<Backend> {
 public:
  explicit inline Erase(const OpSpec &spec)
    : Operator<Backend>(spec)
  {}

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    const auto &input = ws.template InputRef<CPUBackend>(0);
    auto shape = input.shape();
    auto layout = input.GetLayout();
    auto type = input.type();

    //if (spec_->HasTensorArgument("anchor")) {
    //  UseInputAsParams(ws->ArgumentInput("anchor"));
    //} else {
    auto roi_anchor = spec_.template GetArgument<std::vector<float>>("anchor");
    for (auto &x : roi_anchor) {
      std::cout << " " << x;
    }
    std::cout << "\n";

    auto roi_shape = spec_.template GetArgument<std::vector<float>>("shape");
    for (auto &x : roi_shape) {
      std::cout << " " << x;
    }
    std::cout << "\n";

    std::vector<int> axes;
    if (spec_.HasArgument("axis_names")) {
      for (auto axis_name: spec_.GetArgument<TensorLayout>("axis_names")) {
        int d = layout.find(axis_name);
        DALI_ENFORCE(d >= 0);
        axes.push_back(d);
      }
    } else if (spec_.HasArgument("axes")) {
      axes = spec_.GetArgument<std::vector<int>>("axes");
    } else {
      std::cout << "no axes info, expecting all dimensions except 'D'" << std::endl;
      for (int d = 0; d < shape.size(); d++) {
        if (layout[d] == 'C')
          continue;
        axes.push_back(d);
      }
    }

    std::cout << "axes ";
    for (auto axis: axes) {
      std::cout << " " << axis;
    }
    std::cout << "\n";

    if (roi_anchor.empty()) {
      roi_anchor.resize(roi_shape.size());
    }
    DALI_ENFORCE(roi_anchor.size() == roi_shape.size());
    DALI_ENFORCE(roi_shape.size() % axes.size() == 0);

    output_desc.resize(1);
    output_desc[0] = {shape, input.type()};
    return true;
  }

  using Operator<Backend>::RunImpl;
  void RunImpl(workspace_t<Backend> &ws) override;

  bool CanInferOutputs() const override {
    return true;
  }
 
  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_OPERATORS_ERASE_ERASE_H_
