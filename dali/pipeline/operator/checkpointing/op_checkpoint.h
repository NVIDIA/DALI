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

#ifndef DALI_PIPELINE_OPERATOR_CHECKPOINTING_OP_CHECKPOINT_H_
#define DALI_PIPELINE_OPERATOR_CHECKPOINTING_OP_CHECKPOINT_H_

#include <any>
#include <optional>
#include <string>
#include <vector>

#include "dali/core/access_order.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"

namespace dali {

class OpSpec;

/**
 * @brief Representation of a checkpoint for a single operator.
 */
class OpCheckpoint {
 public:
  DLL_PUBLIC explicit OpCheckpoint(const OpSpec &spec);

  /**
   * @brief Returns name of the corresponding operator. Can be used for validation.
   */
  DLL_PUBLIC const std::string &OperatorName() const;

  template<class T> const T &CheckpointState() const {
    try {
      return std::any_cast<const T &>(state_);
    } catch (const std::bad_any_cast &e) {
      DALI_FAIL(make_string("Specified type of requested checkpoint data (`",
                typeid(T).name(),
                "`) doesn't match the data type saved in checkpoint. ",
                e.what()));
    }
  }

  DLL_PUBLIC std::any &MutableCheckpointState();

  /**
   * @brief Sets the access order of the checkpoint data, synchronizing if necessary.
   *
   * GPU operators saving state asynchronously must set the adequate access order.
  */
  DLL_PUBLIC void SetOrder(AccessOrder order);

 private:
  const std::string operator_name_;
  std::any state_;
  AccessOrder order_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_CHECKPOINTING_OP_CHECKPOINT_H_
