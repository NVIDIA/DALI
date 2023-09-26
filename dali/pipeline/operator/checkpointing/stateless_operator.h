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

#ifndef DALI_PIPELINE_OPERATOR_CHECKPOINTING_STATELESS_OPERATOR_H_
#define DALI_PIPELINE_OPERATOR_CHECKPOINTING_STATELESS_OPERATOR_H_

#include <string>

#include "dali/pipeline/operator/operator.h"

namespace dali {

/**
 * @brief Provides trivial checkpointing implementation for stateless operators.
*/
template <typename Backend>
class StatelessOperator : public Operator<Backend> {
 public:
  inline explicit StatelessOperator(const OpSpec &spec) : Operator<Backend>(spec) {}

  inline ~StatelessOperator() override {}

  void SaveState(OpCheckpoint &cpt, std::optional<cudaStream_t> stream) override {}

  void RestoreState(const OpCheckpoint &cpt) override {}

  std::string SerializeCheckpoint(const OpCheckpoint &cpt) const override { return {}; }

  void DeserializeCheckpoint(OpCheckpoint &cpt, const std::string &data) const override {
    DALI_ENFORCE(data.empty(),
                 "Provided checkpoint contains non-empty data for a stateless operator. "
                 "The checkpoint might come from another pipeline. ");
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_CHECKPOINTING_STATELESS_OPERATOR_H_
