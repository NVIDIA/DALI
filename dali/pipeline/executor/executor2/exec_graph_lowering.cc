// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cassert>
#include <unordered_map>
#include <utility>
#include "dali/pipeline/executor/executor2/exec_graph.h"
#include "dali/pipeline/graph/op_graph2.h"
#include "dali/pipeline/operator/error_reporting.h"

namespace dali {
namespace exec2 {

void ExecGraph::Lower(const graph::OpGraph &def) {
  Invalidate();
  std::unordered_map<const graph::OpNode *, ExecNode *> def2exec(def.OpNodes().size());
  for (const graph::OpNode &op_node : def.OpNodes()) {
    std::unique_ptr<OperatorBase> op;
    try {
      op = InstantiateOperator(op_node.spec);
    } catch (...) {
      PropagateError({std::current_exception(),
                      "Critical error when building pipeline:\n" +
                          GetErrorContextMessage(op_node.spec),
                      "\nCurrent pipeline object is no longer valid."});
    }
    ExecNode *exec_node = AddNode(std::move(op), &op_node);
    def2exec.emplace(&op_node, exec_node);
  }

  auto it_def = def.OpNodes().begin();
  for (ExecNode &exec_node : nodes_) {
    assert(it_def != def.OpNodes().end());
    auto op_node = *it_def++;
    assert(exec_node.outputs.size() == op_node.outputs.size());
    for (int o = 0, nout = op_node.outputs.size(); o < nout; o++) {
      const auto &out = op_node.outputs[o];
      auto dev = out->device;
      for (auto &consumer : out->consumers) {
        auto *exec_con = def2exec[consumer.op];
        assert(exec_con != nullptr);
        Link(&exec_node, o, exec_con, consumer.idx)->device = dev;
      }
      exec_node.outputs[o].device = dev;
    }
  }

  ExecNode *out_node = AddOutputNode();

  int pipe_outs = 0;
  for (auto out : def.Outputs()) {
    auto *data_node = def.GetData(out);
    assert(data_node);
    assert(data_node->pipeline_output);
    assert(data_node->producer.op);
    auto *exec_prod = def2exec[data_node->producer.op];
    assert(exec_prod);
    auto *edge = Link(exec_prod, data_node->producer.idx, out_node, pipe_outs++);
    edge->device = data_node->device;
  }

  Sort();
  Validate();
}

}  // namespace exec2
}  // namespace dali
