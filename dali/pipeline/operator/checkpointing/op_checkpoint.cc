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

#include "dali/pipeline/operator/checkpointing/op_checkpoint.h"
#include "dali/pipeline/operator/op_spec.h"

namespace dali {

OpCheckpoint::OpCheckpoint(const OpSpec &spec)
  : operator_name_(spec.name())
  , order_(AccessOrder::host()) {}

const std::string &OpCheckpoint::OperatorName() const {
  return operator_name_;
}

std::any &OpCheckpoint::MutableCheckpointState() {
  return state_;
}

void OpCheckpoint::SetOrder(AccessOrder order) {
  DALI_ENFORCE(order, "Resetting order to an empty one is not supported. ");
  order.wait(order_);
  order_ = order;
}

}  // namespace dali
