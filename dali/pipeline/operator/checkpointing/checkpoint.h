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

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

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
  Checkpoint() {}

  void Clear();

  int AddOperator(std::string instance_name);

  std::optional<int> OperatorIdx(std::string_view instance_name) const;

  OpCheckpoint &GetOpCheckpoint(int index);

  const OpCheckpoint &GetOpCheckpoint(int index) const;

  inline OpCheckpoint &GetOpCheckpoint(std::string_view instance_name) {
    const Checkpoint *cthis = this;
    return const_cast<OpCheckpoint &>(cthis->GetOpCheckpoint(instance_name));
  }

  const OpCheckpoint &GetOpCheckpoint(std::string_view instance_name) const;

  void SetIterationId(size_t iteration_id);

  size_t GetIterationId() const;

  /**
   * @brief Sets the given order on all the OpCheckpoints.
   *
   * Can be used to synchronize the checkpoints.
  */
  void SetOrder(AccessOrder order);

  /**
   * @brief Returns the number of OpCheckpoints kept.
   *
   * It's equivalent to the number of operators in the related pipeline.
  */
  Index NumOp() const;

  /**
   * @brief Serializes this entire object into a serialized protobuf message.
  */
  std::string SerializeToProtobuf(ExecutorBase &exec) const;

  /**
   * @brief Deserializes a protobuf message and builds this object.
  */
  void DeserializeFromProtobuf(ExecutorBase &exec, std::string_view serialized_data);

  ExternalContextCheckpoint external_ctx_cpt_{};

 private:
  std::vector<OpCheckpoint> cpts_;
  std::map<std::string, int, std::less<>> name2id_;
  size_t iteration_id_ = 0;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_CHECKPOINTING_CHECKPOINT_H_
