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

#ifndef DALI_PIPELINE_EXECUTOR_EXECUTOR2_OP_TASK_H_
#define DALI_PIPELINE_EXECUTOR_EXECUTOR2_OP_TASK_H_

#include <memory>
#include <utility>
#include "dali/core/exec/tasking.h"
#include "dali/core/call_at_exit.h"
#include "dali/pipeline/executor/executor2/shared_event_lease.h"
#include "dali/pipeline/executor/executor2/exec_graph.h"

namespace dali {
namespace exec2 {

class ExecNode;

template <typename Backend>
struct OperatorIO {
  const std::shared_ptr<TensorList<Backend>> data;
  SharedEventLease event;
  AccessOrder order = AccessOrder::host();
};

struct PipelineOutput;

/** A context for task functions.
 *
 * OpTask is a context that's passed (by move) to the runnable used in tasking::Task.
 */
class OpTask {
 public:
  OpTask(OpTask &&) = default;
  OpTask(const OpTask &) {
    std::cerr << "This constructor is here only because std::function requires "
                 "the functor to be copy-constructible. We never actually copy the target.\n"
                 "See C++23 std::move_only_function." << std::endl;
    std::abort();
  }

  /** Creates a tasking::Task from an ExecNode and a Workspace.
   *
   * There are two possible tasks:
   * - operator task
   * - output task.
   */
  static tasking::SharedTask CreateTask(ExecNode *node, const WorkspaceParams &params);

 private:
  OpTask(ExecNode *node, WorkspaceParams ws_params)
  : node_(node), ws_params_(std::move(ws_params)) {}

  /** Gets a functor that returns an output Workspace compatible with DALI pipeline.
   */
  auto GetOutputTaskRunnable() && {
    assert(node_->is_pipeline_output);
    return [self = std::move(*this)](tasking::Task *t) mutable {
      self.task_ = t;
      return self.GetOutput();
    };
  }

  /** Gets a functor that runs the operator. */
  auto GetOpTaskRunnable() && {
    assert(!node_->is_pipeline_output);
    return [self = std::move(*this)](tasking::Task *t) mutable {
      self.task_ = t;
      return self.Run();
    };
  }

  using OpTaskOutputs = SmallVector<std::any, 8>;

  OpTaskOutputs Run();

  tasking::Task *task_ = nullptr;
  ExecNode *node_ = nullptr;
  WorkspaceParams ws_params_{};
  std::unique_ptr<Workspace> ws_ = nullptr;
  SharedEventLease event_;
  /** If true, the operator's Setup and Run are skipped. */
  bool skip_ = false;

  template <typename Backend>
  const auto &TaskInput(int i) const {
    return task_->GetInputValue<const OperatorIO<Backend> &>(i);
  }

  void SetWorkspaceInputs();
  void SetupOp();
  void RunOp();
  OpTaskOutputs GetWorkspaceOutputs();

  /** Returns a workspace in DALI pipeline compatible format (along with supporting structures).
   *
   * In case of operators, the inputs of the node become inputs of the workspace. In case of
   * pipeline output, the inputs of the output node become the _outputs_ of the workspace.
   */
  PipelineOutput GetOutput();

  /** Returns the order in which the output is consumed iff the consumption order is uniform.
   *
   * This function is used for optimizing stream transition in cases where all consumers
   * work on one stream. When the output is consumed on multiple streams, the result is empty
   * and the stream assignment of the output is not updated.
   */
  AccessOrder OutputConsumerOrder(int output_idx);

  /** Checks if the execution of the operator should be skipped.
   *
   * The operator is skipped if:
   * 1. It has inputs and all of them are empty batches.
   * 2. It has no inputs and the requested batch size is 0.
   */
  bool ShouldSkip() const;

  void ApplyDefaultLayouts();

  /** Applies a default layout to the input at index input_idx
   *
   * Operators have default layouts for inputs of specific rank, e.g. HWC for 3D.
   * When no layout is specified, this function may adjust it.
   */
  template <typename Backend>
  void ApplyDefaultLayout(int input_idx, const OpSchema &schema);

  void ResetInputLayouts();

  void ClearWorkspace();

  auto GetWorkspace() {
    assert(!ws_);
    std::tie(ws_, event_) = node_->GetWorkspace(ws_params_);
    return AtScopeExit([this]() {
      ClearWorkspace();
      node_->PutWorkspace(std::move(ws_));
    });
  }

  SmallVector<int, 4> reset_input_layouts_;
};


}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR2_OP_TASK_H_
