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

#ifndef DALI_PIPELINE_OPERATOR_BUILTIN_CONDITIONAL_MERGE_H_
#define DALI_PIPELINE_OPERATOR_BUILTIN_CONDITIONAL_MERGE_H_

#include <optional>
#include <vector>

#include "dali/core/access_order.h"
#include "dali/core/common.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class Merge : public StatelessOperator<Backend> {
 public:
  explicit Merge(const OpSpec &spec) : StatelessOperator<Backend>(spec) {
    DALI_ENFORCE(spec.HasTensorArgument("predicate"),
                 "The 'predicate' argument is required to be present as argument input.");
    RegisterTestsDiagnostics();
  }

  ~Merge() override = default;

  bool CanInferOutputs() const override {
    return false;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;
  void RunImpl(Workspace &ws) override;

  DISABLE_COPY_MOVE_ASSIGN(Merge);

 private:
  /**
   * @brief Fallback for scheduling copy in a thread pool or on a stream
   */
  void CopySampleToOutput(TensorList<Backend> &output, int output_idx,
                          const TensorList<Backend> &input, int input_idx,
                          Workspace &ws);

  /**
   * @brief For CPU backend, execute the work scheduled in thread pool.
   */
  void FinalizeCopy(Workspace &ws);

  void RegisterTestsDiagnostics();
  void WriteTestsDiagnostics(const Workspace &ws);

  USE_OPERATOR_MEMBERS();

  // We can only merge two batches based on a boolean predicate.
  static constexpr int kMaxGroups = 2;
  int input_sample_count_ = 0;
  std::optional<bool> pinned_;
  int device_id_ = CPU_ONLY_DEVICE_ID;
  std::optional<AccessOrder> order_;

  // test diagnostics
  bool in_0_pinned_, in_1_pinned_, out_pinned_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_CONDITIONAL_MERGE_H_
