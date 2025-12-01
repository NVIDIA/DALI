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

#ifndef DALI_PIPELINE_EXECUTOR_EXECUTOR_H_
#define DALI_PIPELINE_EXECUTOR_EXECUTOR_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "dali/core/common.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/pipeline/operator/checkpointing/checkpoint.h"
#include "dali/pipeline/graph/op_graph2.h"

namespace dali {

class OperatorBase;

struct DLL_PUBLIC ExecutorMeta {
  size_t real_size;
  size_t max_real_size;
  size_t reserved;
  size_t max_reserved;
};

using ExecutorMetaMap = std::unordered_map<std::string, std::vector<ExecutorMeta>>;

class OpGraph;

class DLL_PUBLIC ExecutorBase {
 public:
  DLL_PUBLIC virtual ~ExecutorBase() {}
  DLL_PUBLIC virtual void Build(const graph::OpGraph &graph) = 0;
  // TODO(michalz): Remove
  DLL_PUBLIC virtual void Build(OpGraph *graph, std::vector<std::string> output_names) = 0;
  DLL_PUBLIC virtual void Init() = 0;
  DLL_PUBLIC virtual void Run() = 0;
  DLL_PUBLIC virtual void Prefetch() = 0;
  DLL_PUBLIC virtual void Outputs(Workspace *ws) = 0;
  DLL_PUBLIC virtual void ShareOutputs(Workspace *ws) = 0;
  DLL_PUBLIC virtual void ReleaseOutputs() = 0;
  DLL_PUBLIC virtual void EnableMemoryStats(bool enable_memory_stats = false) = 0;
  DLL_PUBLIC virtual void EnableCheckpointing(bool checkpointing = false) = 0;
  DLL_PUBLIC virtual ExecutorMetaMap GetExecutorMeta() = 0;
  DLL_PUBLIC virtual void Shutdown() = 0;
  DLL_PUBLIC virtual Checkpoint& GetCurrentCheckpoint() = 0;
  DLL_PUBLIC virtual void RestoreStateFromCheckpoint(const Checkpoint &cpt) = 0;
  DLL_PUBLIC virtual int InputFeedCount(std::string_view input_name) = 0;
  DLL_PUBLIC virtual OperatorBase *GetOperator(std::string_view name) = 0;

 protected:
  /**
   * @brief Returns true if conditionals are used in the executed graph, @see DetectConditionals().
   * Valid after Build().
   */
  virtual bool HasConditionals() const = 0;

  template <typename T>
  friend class ExecutorTest;
};

}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR_H_
