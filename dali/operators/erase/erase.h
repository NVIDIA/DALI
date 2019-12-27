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

#include <memory>
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

namespace detail {

template <typename Backend>
class OpImplBase {
 public:
  virtual ~OpImplBase() = default;
  virtual bool SetupImpl(std::vector<OutputDesc> &output_desc,
                         const workspace_t<Backend> &ws) = 0;
  virtual void RunImpl(workspace_t<Backend> &ws) = 0;
};

std::vector<int> GetAxes(const OpSpec &spec, TensorLayout layout) {
  std::vector<int> axes;
  if (spec.HasArgument("axis_names")) {
    for (auto axis_name : spec.GetArgument<TensorLayout>("axis_names")) {
      int d = layout.find(axis_name);
      DALI_ENFORCE(d >= 0);
      axes.push_back(d);
    }
  } else if (spec.HasArgument("axes")) {
    axes = spec.GetArgument<std::vector<int>>("axes");
  } else {
    // no axes info, expecting all dimensions except 'C'
    for (int d = 0; d < layout.size(); d++) {
      if (layout[d] == 'C')
        continue;
      axes.push_back(d);
    }
  }
  return axes;
}

}  // namespace detail

template <typename Backend>
class Erase : public Operator<Backend> {
 public:
  explicit inline Erase(const OpSpec &spec)
    : Operator<Backend>(spec)
    , spec__(spec) {}

 protected:
  using Operator<Backend>::RunImpl;
  void RunImpl(workspace_t<Backend> &ws) override;
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override;

  bool CanInferOutputs() const override {
    return true;
  }

  USE_OPERATOR_MEMBERS();

  OpSpec spec__;
  std::unique_ptr<detail::OpImplBase<Backend>> impl_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_ERASE_ERASE_H_
