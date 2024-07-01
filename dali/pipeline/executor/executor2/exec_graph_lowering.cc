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
#include "dali/pipeline/graph/op_graph2.h"

namespace dali {
namespace exec2 {

void ExecGraph::Lower(const graph::OpGraph &def) {
  for (const graph::OpNode &op_node : def.OpNodes()) {
    ExecNode *exec_node = AddNode(InstantiateOperator(op_node.spec), &op_node);
    def2exec.emplace(&op_node, exec_node);
  }

  for (ExecNode &exec_node : nodes) {
    auto *op_node = exec_node.def;
    for (int o = 0, nout = op_node->outputs.size(); o < nout; o++) {
      const auto &out = op_node->outputs[o];
      for (auto &consumer : out->consumers) {
        auto *exec_con = def2exec[consumer.op];
        assert(exec_con != nullptr);
        Link(&exec_node, o, exec_con, consumer.idx);
      }
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
    Link(exec_prod, data_node->producer.idx, out_node, pipe_outs++);
  }
}

}  // namespace exec2
}  // namespace dali
