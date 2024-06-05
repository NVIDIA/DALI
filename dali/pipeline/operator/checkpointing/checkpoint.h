// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string>

#include "dali/pipeline/operator/checkpointing/op_checkpoint.h"

namespace dali {

class ExecutorBase;

/**
 * @brief Pipeline-wide state, passed from python side
*/
struct ExternalContextCheckpoint {
  std::string pipeline_data;
  std::string iterator_data;
};

/**
 * @brief Aggregation of operator checkpoints for a whole pipeline.
 */
class DLL_PUBLIC Checkpoint {
 public:
  DLL_PUBLIC Checkpoint() {}

  DLL_PUBLIC void Clear();

  DLL_PUBLIC int AddOperator(std::string instance_name);

  DLL_PUBLIC OpCheckpoint &GetOpCheckpoint(int index);

  DLL_PUBLIC const OpCheckpoint &GetOpCheckpoint(int index) const;

  DLL_PUBLIC void SetIterationId(size_t iteration_id);

  DLL_PUBLIC size_t GetIterationId() const;

  /**
   * @brief Sets the given order on all the OpCheckpoints.
   *
   * Can be used to synchronize the checkpoints.
  */
  DLL_PUBLIC void SetOrder(AccessOrder order);

  /**
   * @brief Returns the number of OpCheckpoints kept.
   *
   * It's equivalent to the number of operators in the related pipeline.
  */
  DLL_PUBLIC Index NumOp() const;

  /**
   * @brief Serializes this entire object into a serialized protobuf message.
  */
  DLL_PUBLIC std::string SerializeToProtobuf(ExecutorBase &exec) const;

  /**
   * @brief Deserializes a protobuf message and builds this object.
  */
  DLL_PUBLIC void DeserializeFromProtobuf(ExecutorBase &exec,
                                          const std::string &serialized_data);

  ExternalContextCheckpoint external_ctx_cpt_{};

 private:
  std::vector<OpCheckpoint> cpts_;
  size_t iteration_id_ = 0;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_CHECKPOINTING_CHECKPOINT_H_
