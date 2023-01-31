// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_CONDITIONAL_UNARY_H_
#define DALI_PIPELINE_OPERATOR_CONDITIONAL_UNARY_H_

#include <string>

#include "dali/pipeline/operator/operator.h"

namespace dali {

/**
 * @brief Base class for implementing the logical ``and`` and ``or`` operators.
 * Handles the type and shape validation of inputs.
 */
class LogicalNot : public Operator<CPUBackend> {
 public:
  explicit LogicalNot(const OpSpec &spec) : Operator<CPUBackend>(spec), name_("not") {}

  ~LogicalNot() override = default;

  bool CanInferOutputs() const override {
    return false;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;
  void RunImpl(Workspace &ws) override;

  DISABLE_COPY_MOVE_ASSIGN(LogicalNot);

 private:
  USE_OPERATOR_MEMBERS();

  std::string name_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_CONDITIONAL_UNARY_H_
