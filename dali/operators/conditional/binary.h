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

#ifndef DALI_PIPELINE_OPERATOR_CONDITIONAL_BINARY_H_
#define DALI_PIPELINE_OPERATOR_CONDITIONAL_BINARY_H_

#include <string>

#include "dali/pipeline/operator/operator.h"

namespace dali {

/**
 * @brief Base class for implementing the logical ``and`` and ``or`` operators.
 * Handles the type and shape validation of inputs.
 */
class BinaryLogicalOp : public Operator<CPUBackend> {
 public:
  explicit BinaryLogicalOp(const OpSpec &spec, std::string name)
      : Operator<CPUBackend>(spec), name_(std::move(name)) {}

  ~BinaryLogicalOp() override = default;

  bool CanInferOutputs() const override {
    return false;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;
  void RunImpl(Workspace &ws) override;

  /**
   * @brief Used to provide implementation of specific logical operation.
   */
  virtual bool compute(bool left, bool right) const = 0;

  DISABLE_COPY_MOVE_ASSIGN(BinaryLogicalOp);

 private:
  USE_OPERATOR_MEMBERS();

  std::string name_;
};

class LogicalAnd : public BinaryLogicalOp {
 public:
  explicit LogicalAnd(const OpSpec &spec) : BinaryLogicalOp(spec, "and") {}
  bool compute(bool left, bool right) const override {
    return left && right;
  }
};

class LogicalOr : public BinaryLogicalOp {
 public:
  explicit LogicalOr(const OpSpec &spec) : BinaryLogicalOp(spec, "or") {}
  bool compute(bool left, bool right) const override {
    return left || right;
  }
};


}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_CONDITIONAL_BINARY_H_
