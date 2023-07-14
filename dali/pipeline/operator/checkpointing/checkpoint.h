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

#ifndef DALI_PIPELINE_OPERATOR_CHECKPOINTING_CHECKPOINT_H_
#define DALI_PIPELINE_OPERATOR_CHECKPOINTING_CHECKPOINT_H_

#include <vector>

#include "dali/pipeline/graph/op_graph.h"

namespace dali {

/**
 * @brief Aggregation of operator checkpoints for a whole pipeline.
 */
class DLL_PUBLIC Checkpoint {
 public:
  DLL_PUBLIC Checkpoint() {}

  DLL_PUBLIC void Build(const OpGraph &graph);

  DLL_PUBLIC OpCheckpoint &GetOpCheckpoint(OpNodeId id);

  DLL_PUBLIC const OpCheckpoint &GetOpCheckpoint(OpNodeId id) const;

 private:
  std::vector<OpCheckpoint> cpts_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_CHECKPOINTING_CHECKPOINT_H_
