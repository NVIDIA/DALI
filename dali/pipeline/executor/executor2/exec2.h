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
#include "dali/pipeline/executor/executor.h"

namespace dali {
namespace exec2 {

class DLL_PUBLIC Executor2 : public ExecutorBase {
 public:
  explicit Executor2(int queue_depth);

  void Build(const graph::OpGraph &graph) override {

  }
  // TODO(michalz): Remove
  void Build(OpGraph *graph, std::vector<std::string> output_names) override {
    throw std::logic_error("This function is maintained in the interface for legacy tests only.");
  }

  void Init() override;
  void Run() override;
  void Prefetch() override;

  void Outputs(Workspace *ws) override;
  void ShareOutputs(Workspace *ws) override;
  void ReleaseOutputs() override;
  void EnableMemoryStats(bool enable_memory_stats = false) override;
  void EnableCheckpointing(bool checkpointing = false) override;
  ExecutorMetaMap GetExecutorMeta() override;
  void Shutdown() override;
  Checkpoint& GetCurrentCheckpoint() override;
  void RestoreStateFromCheckpoint(const Checkpoint &cpt) override;
  int InputFeedCount(std::string_view input_name) override;
  OperatorBase *GetOperator(std::string_view name) override;

  void Initialize(std::shared_ptr<graph::OpGraph> graph) {
    graph_ = graph;
  }

  ExecGraph exec_graph_;
  std::shared_ptr<graph::OpGraph> graph_;
};

}  // namespace exec2
}  // namespace dali


#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC2_H_
