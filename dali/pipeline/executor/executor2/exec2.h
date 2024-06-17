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

#ifndef DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC2_H_
#define DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC2_H_

#include <memory>
#include "dali/pipeline/graph/op_graph2.h"
#include "dali/pipeline/executor/executor2/exec_graph.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {
namespace exec2 {

class Executor2 {
 public:
  void Initialize(std::shared_ptr<graph::OpGraph> graph) {
    graph_ = graph;
  }

  void Run() {
  }

  void GetOutputs(Workspace &ws) {
  }

  ExecGraph exec_graph_;
  std::shared_ptr<graph::OpGraph> graph_;
};

}  // namespace exec2
}  // namespace dali


#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC2_H_
