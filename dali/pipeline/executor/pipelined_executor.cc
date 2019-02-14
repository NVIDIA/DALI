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
#include "dali/pipeline/operators/common.h"

namespace dali {

void PipelinedExecutor::Build(OpGraph *graph, vector<string> output_names) {
  Executor::Build(graph, output_names);
}

void PipelinedExecutor::SetupOutputInfo(const OpGraph &graph) {
  DeviceGuard g(device_id_);
  Executor::SetupOutputInfo(graph);
  constexpr auto stages_count = static_cast<int>(DALIOpType::COUNT);
  stage_outputs_.resize(stages_count);
  // stage_output_events_.resize(stages_count);
  for (int stage = 0; stage < stages_count; stage++) {
    stage_outputs_[stage] = graph.GetStageOutputs(static_cast<DALIOpType>(stage));
    // for (auto tid : stage_outputs_[stage]) {

    // }
  }
}

std::vector<int> PipelinedExecutor::GetTensorQueueSizes(const OpGraph &graph) {
  std::vector<int> result;
  result.resize(graph.NumTensor(), 1);
  auto output_ids = graph.GetOutputs(output_names_);
  for (int stage = 0; stage < static_cast<int>(DALIOpType::COUNT); stage++) {
    auto stage_outputs = graph.GetStageOutputs(static_cast<DALIOpType>(stage));
    output_ids.insert(output_ids.end(), stage_outputs.begin(), stage_outputs.end());
  }
  for (auto id : output_ids) {
    result[id] = queue_depth_;
  }
  return result;
}

}  // namespace dali

