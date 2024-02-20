// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_BUILTIN_CONDITIONAL_SPLIT_H_
#define DALI_PIPELINE_OPERATOR_BUILTIN_CONDITIONAL_SPLIT_H_

#include <vector>

#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/operator.h"


namespace dali {

template <typename Backend>
class Split : public StatelessOperator<Backend> {
 public:
  explicit Split(const OpSpec &spec)
      : StatelessOperator<Backend>(spec),
        if_stmt_implementation_(spec.GetArgument<bool>("_if_stmt")) {
    DALI_ENFORCE(spec.HasTensorArgument("predicate"),
                 "The 'predicate' argument is required to be present as argument input.");
    RegisterTestsDiagnostics();
  }

  ~Split() override = default;

  bool CanInferOutputs() const override {
    return false;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;
  void RunImpl(Workspace &ws) override;

  DISABLE_COPY_MOVE_ASSIGN(Split);

 private:
  void RegisterTestsDiagnostics();
  void WriteTestsDiagnostics(const Workspace &ws);

  USE_OPERATOR_MEMBERS();

  // We can only split two batches based on a boolean predicate.
  static constexpr int kMaxGroups = 2;
  std::array<int, kMaxGroups> group_counts_;

  bool if_stmt_implementation_ = false;

  // test diagnostics
  bool in_pinned_, out_0_pinned_, out_1_pinned_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_CONDITIONAL_SPLIT_H_
