// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include <algorithm>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "dali/pipeline/executor/pipelined_executor.h"
#include "dali/pipeline/operator/common.h"

namespace dali {
template<typename WorkspacePolicy, typename QueuePolicy>
size_t PipelinedExecutorImpl<WorkspacePolicy, QueuePolicy>::CalcIterationDataSize() const {
  // TODO
  DALI_ENFORCE(this->queue_sizes_.cpu_size == this->queue_sizes_.gpu_size);
  return this->queue_sizes_.cpu_size+1;
//  return this->queue_sizes_.cpu_size + this->queue_sizes_.gpu_size +
//         this->queue_sizes_.gpu_size /* mixed_size is equal to gpu_size */;
}


template<typename WorkspacePolicy, typename QueuePolicy>
IterationData &
PipelinedExecutorImpl<WorkspacePolicy, QueuePolicy>::GetCurrentIterationData(
        size_t iteration_id, OpType op_type) {
  // TODO
//  assert(op_type == OpType::CPU || op_type == OpType::GPU || op_type == OpType::MIXED);
//  switch(op_type) {
//    case OpType::CPU:
//      return this->iteration_data_[iteration_id % (this->queue_sizes_.cpu_size + 1)];
//    case OpType::MIXED:
//      return 9;
//    case OpType::GPU:
//      return 0;
//    default:
//      DALI_FAIL("This part of code should be unreachable.");
//  }
  return this->iteration_data_[iteration_id %
                               (op_type == OpType::CPU ? this->queue_sizes_.cpu_size
                                                       : this->queue_sizes_.gpu_size)];
}


template class DLL_PUBLIC PipelinedExecutorImpl<AOT_WS_Policy<UniformQueuePolicy>, UniformQueuePolicy>;
template class DLL_PUBLIC PipelinedExecutorImpl<AOT_WS_Policy<SeparateQueuePolicy>, SeparateQueuePolicy>;

}  // namespace dali
