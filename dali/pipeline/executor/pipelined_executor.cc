// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/executor/pipelined_executor.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

template<typename WorkspacePolicy, typename QueuePolicy>
size_t PipelinedExecutorImpl<WorkspacePolicy, QueuePolicy>::CalcIterationDataSize() const {
  /*
   * With PipelinedExecutor (with both Uniform and Separated queues), CPU, Mixed and GPU stages
   * can run simultaneously as many ops as their queue sizes. In other words, in the CPU stage
   * there can be `cpu_size` iterations alive, etc. Moreover, cpu_size > gpu_size.
   *
   * Therefore, the total number of required iteration data structs is a sum of the stages plus one.
   * This one is required for the output Workspace.
   */
  return this->queue_sizes_.cpu_size + this->queue_sizes_.gpu_size +
         this->queue_sizes_.gpu_size /* mixed_queue_size */ + 1;
}

int SeparatedPipelinedExecutor::InputFeedCount(std::string_view op_name) {
  (void)graph_->Node(op_name);
  return queue_sizes_.cpu_size + queue_sizes_.gpu_size;
}

template
class DLL_PUBLIC PipelinedExecutorImpl<AOT_WS_Policy<UniformQueuePolicy>, UniformQueuePolicy>;

}  // namespace dali
