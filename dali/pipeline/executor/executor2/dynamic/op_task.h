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

#ifndef DALI_PIPELINE_EXECUTOR2_DYNAMIC_OP_TASK_H_
#define DALI_PIPELINE_EXECUTOR2_DYNAMIC_OP_TASK_H_

#include <memory>
#include "workspace_cache.h"
#include "dali/core/exec/tasking.h"

namespace dali {
namespace exec2 {

class ExecNode;

class OpTaskFunc {
 private:
  OpTaskFunc(ExecNode *node, CachedWorkspace ws)
  : node_(node), ws_(std::move(ws)) {}

  auto GetOutputTaskRunnable() &&;
  auto GetOpTaskRunnable() &&;


 public:
  OpTaskFunc(OpTaskFunc &&) = default;
  OpTaskFunc(const OpTaskFunc &) {
    std::cerr << "This constructor is here only because std::function requires "
                 "the functor to be copy-constructible. We never actually copy the target.\n"
                 "See C++23 std::move_only_function." << std::endl;
    std::abort();
  }

  static tasking::SharedTask CreateTask(ExecNode *node, CachedWorkspace ws);

 private:
  using OpTaskOutputs = SmallVector<std::any, 8>;

  OpTaskOutputs Run();

  tasking::Task *task_ = nullptr;
  ExecNode *node_ = nullptr;
  CachedWorkspace ws_;

  template <typename Backend>
  const auto &TaskInput(int i) const {
    return task_->GetInputValue<const std::shared_ptr<TensorList<Backend>> &>(i);
  }

  void SetWorkspaceInputs();
  void SetupOp();
  void RunOp();
  void ResetWorkspaceInputs();
  OpTaskOutputs GetWorkspaceOutputs();
  Workspace GetOutput();
};


}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR2_DYNAMIC_OP_TASK_H_
