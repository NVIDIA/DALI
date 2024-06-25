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
#include <utility>
#include "dali/pipeline/executor/executor2/exec_graph.h"

namespace dali {
namespace exec2 {

void ExecGraph::Lower(const graph::OpGraph &def) {
  for (const graph::OpNode &op_node : def.OpNodes()) {
    ExecNode *exec_node = AddNode(InstantiateOperator(op_node.spec), &op_node);
    def2exec.emplace(&op_node, exec_node);
  }
  ExecNode *out_node = AddOutputNode();

  std::map<std::string_view, ExecEdge *, std::less<>> output_map;

  for (ExecNode &exec_node : nodes) {
    auto *op_node = exec_node.def;
    for (int o = 0, nout = op_node->outputs.size(); o < nout; o++) {
      auto &edge = edges.emplace_back();
      edge.producer = &exec_node;
      edge.producer_output_idx = o;
      const auto &out = op_node->outputs[o];
      if (out->pipeline_output) {
        bool inserted = output_map.emplace(out->name, &edge).second;
        (void)inserted;
        assert(inserted);
      }
    }
  }

  for (ExecNode &exec_node : nodes) {
    auto *op_node = exec_node.def;
    for (int i = 0, ninp = op_node->inputs.size(); i < ninp; i++) {
      const auto &inp = op_node->inputs[i];
      auto *exec_prod = def2exec[inp->producer.op];
      assert(exec_prod != nullptr);
      auto *edge = exec_prod->outputs.at(inp->producer.idx);
      edge->consumer = &exec_node;
      edge->consumer_input_idx = i;
      exec_node.inputs.push_back(edge);
    }
  }

  for (auto out : def.Outputs()) {
    auto *edge = output_map[out];
    assert(edge != nullptr);
    out_node->outputs.push_back(edge);
  }
}

}  // namespace exec2
}  // namespace dali
