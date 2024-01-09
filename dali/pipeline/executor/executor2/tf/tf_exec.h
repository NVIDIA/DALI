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

#ifndef DALI_PIPELINE_EXECUTOR2_TF_TF_EXEC_H_
#define DALI_PIPELINE_EXECUTOR2_TF_TF_EXEC_H_

#include "../graph.h"
#include "../executor2.h"
#include "exec_graph.h"

#include "third_party/taskflow/taskflow/taskflow.hpp"  // TODO(michalz): Add it to cmake

namespace dali {
namespace exec2 {

class TFExec : public Executor {
 public:
  void Initialize(std::shared_ptr<Graph> graph) override
  {
    graph_ = graph;
  }

  void Run() override
  {
  }

  void GetOutputs(Workspace &ws) override
  {
  }

  ExecGraph exec_graph_;
  std::shared_ptr<Graph> graph_;
};

}  // namespace exec2
}  // namespace dali


#endif  // DALI_PIPELINE_EXECUTOR2_TF_TF_EXEC_H_

