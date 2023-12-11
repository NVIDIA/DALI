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

#ifndef DALI_PIPELINE_OPERATOR_BUILTIN_CONDITIONAL_VALIDATE_LOGICAL_EXPR_H_
#define DALI_PIPELINE_OPERATOR_BUILTIN_CONDITIONAL_VALIDATE_LOGICAL_EXPR_H_

#include <string>
#include <vector>

#include "dali/pipeline/operator/builtin/conditional/validation.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

/**
 * @brief Operator used for checking the input type and shape for lazy logical expressions `or`
 * and `and`. The inputs are restricted to scalars, it passes them through, but copy should also
 * be a similarly valid option.
 */
class LogicalValidate : public StatelessOperator<CPUBackend> {
 public:
  explicit LogicalValidate(const OpSpec &spec)
      : StatelessOperator<CPUBackend>(spec),
        name_(spec.GetArgument<std::string>("expression_name")),
        side_(spec.GetArgument<std::string>("expression_side")) {}

  ~LogicalValidate() override = default;

  bool CanInferOutputs() const override {
    return false;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;
  void RunImpl(Workspace &ws) override;

  DISABLE_COPY_MOVE_ASSIGN(LogicalValidate);

 private:
  USE_OPERATOR_MEMBERS();

  std::string name_;
  std::string side_;
};

/**
 * @brief This is just a placeholder operator that is picked when GPU inputs are encountered
 * and reports a better error.
 */
class LogicalFailForGpu : public StatelessOperator<GPUBackend> {
 public:
  explicit LogicalFailForGpu(const OpSpec &spec)
      : StatelessOperator<GPUBackend>(spec),
        name_(spec.GetArgument<std::string>("expression_name")),
        side_(spec.GetArgument<std::string>("expression_side")) {
    ReportGpuInputError(name_, side_, true);
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    return false;
  }

  void RunImpl(Workspace &ws) override {}

  ~LogicalFailForGpu() override = default;

  DISABLE_COPY_MOVE_ASSIGN(LogicalFailForGpu);

 private:
  USE_OPERATOR_MEMBERS();

  std::string name_;
  std::string side_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_CONDITIONAL_VALIDATE_LOGICAL_EXPR_H_
