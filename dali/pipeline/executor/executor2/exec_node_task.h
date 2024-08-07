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

#ifndef DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC_NODE_TASK_H_
#define DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC_NODE_TASK_H_

#include <memory>
#include <utility>
#include "dali/core/exec/tasking.h"
#include "dali/core/call_at_exit.h"
#include "dali/pipeline/executor/executor2/shared_event_lease.h"
#include "dali/pipeline/executor/executor2/exec_graph.h"

namespace dali {
namespace exec2 {

class ExecNode;

/** A context for task functions.
 *
 * ExecNodeTask is a context that's passed (by move) to the runnable used in tasking::Task.
 */
class ExecNodeTask {
 public:
  ExecNodeTask(ExecNodeTask &&) = default;
  ExecNodeTask(const ExecNodeTask &) {
    std::cerr << "This constructor is here only because std::function requires "
                 "the functor to be copy-constructible. We never actually copy the target.\n"
                 "See C++23 std::move_only_function." << std::endl;
    std::abort();
  }

  /** Creates a tasking::Task from an ExecNode and WorkspaceParams.
   *
   * There are two possible tasks:
   * - operator task
   * - output task.
   */
  static tasking::SharedTask CreateTask(ExecNode *node, const WorkspaceParams &params);

 protected:
  ExecNodeTask(ExecNode *node, WorkspaceParams ws_params)
  : node_(node), ws_params_(std::move(ws_params)) {}

  tasking::Task *task_ = nullptr;
  ExecNode *node_ = nullptr;
  WorkspaceParams ws_params_{};
  std::unique_ptr<Workspace> ws_ = nullptr;
  SharedEventLease event_;

  template <typename Backend>
  struct OperatorIO {
    const std::shared_ptr<TensorList<Backend>> data;
    SharedEventLease event;
    AccessOrder order = AccessOrder::host();
  };

  template <typename Backend>
  const auto &TaskInput(int i) const {
    return task_->GetInputValue<const OperatorIO<Backend> &>(i);
  }

  /** Prepares the workspace for recycling */
  void ClearWorkspace();

  auto GetWorkspace() {
    assert(!ws_);
    std::tie(ws_, event_) = node_->GetWorkspace(ws_params_);
    return AtScopeExit([this]() {
      ClearWorkspace();
      node_->PutWorkspace(std::move(ws_));
    });
  }
};


}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR2_EXEC_NODE_TASK_H_
