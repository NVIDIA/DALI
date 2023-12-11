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

#ifndef DALI_PIPELINE_OPERATOR_BUILTIN_CONDITIONAL_LOGICAL_NOT_H_
#define DALI_PIPELINE_OPERATOR_BUILTIN_CONDITIONAL_LOGICAL_NOT_H_

#include <string>
#include <vector>

#include "dali/pipeline/operator/builtin/conditional/validation.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/operator.h"


namespace dali {

/**
 * @brief Eager `not` operator from Python
 */
class LogicalNot : public StatelessOperator<CPUBackend> {
 public:
  explicit LogicalNot(const OpSpec &spec) : StatelessOperator<CPUBackend>(spec), name_("not") {}

  ~LogicalNot() override = default;

  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;
  void RunImpl(Workspace &ws) override;

  DISABLE_COPY_MOVE_ASSIGN(LogicalNot);

 private:
  USE_OPERATOR_MEMBERS();

  std::string name_;
};

/**
 * @brief This is just a placeholder operator that is picked when GPU inputs are encountered
 * and reports a better error.
 */
class LogicalNotFailForGpu : public Operator<GPUBackend> {
 public:
  explicit LogicalNotFailForGpu(const OpSpec &spec) : Operator<GPUBackend>(spec) {
    ReportGpuInputError("not", "", true);
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    return false;
  }

  void RunImpl(Workspace &ws) override {}

  ~LogicalNotFailForGpu() override = default;

  DISABLE_COPY_MOVE_ASSIGN(LogicalNotFailForGpu);

 private:
  USE_OPERATOR_MEMBERS();
};


}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_CONDITIONAL_LOGICAL_NOT_H_
