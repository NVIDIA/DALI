
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
#include <string>
#include <vector>

#include "dali/pipeline/operator/op_spec.h"
#include "dali/core/access_order.h"

namespace dali {

/**
 * @brief Representation of a checkpoint for a single operator.
 */
class OpCheckpoint {
 public:
  OpCheckpoint(const OpSpec &spec) : operator_name_(spec.name()) {}

  const std::string &OperatorName() const {
    return operator_name_;
  }

  template<class T> const T &CheckpointState() const {
    try {
      return std::any_cast<const T &>(state_);
    } catch (const std::bad_any_cast &e) {
      DALI_FAIL(make_string("Specified type of requested checkpoint data ` (`", 
                typeid(T).name(),
                "`) doesn't match the data type saved in checkpoint. ",
                e.what()));
    }
  }

  std::any &MutableCheckpointState() {
    return state_;
  }

  AccessOrder& Order() { return order_; }

 private:
  AccessOrder order_;
  std::any state_;
  const std::string operator_name_;  // used for a sanity check
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_CHECKPOINTING_CHECKPOINT_H_