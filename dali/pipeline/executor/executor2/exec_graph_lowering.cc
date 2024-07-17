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

namespace dali {
namespace exec2 {

void ExecGraph::Lower(const graph::OpGraph &def) {
  std::unordered_map<const graph::OpNode *, ExecNode *> def2exec(def.OpNodes().size());
  for (const graph::OpNode &op_node : def.OpNodes()) {
    ExecNode *exec_node = AddNode(InstantiateOperator(op_node.spec), &op_node);
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

  FindPinnedBuffers();
  Validate();
}

namespace {

/** Sets pinnedness of the input sources
 *
 * The function goes over the inputs of the node. If the node is non-CPU, then all of its
 * CPU _regular_ inputs are marked as pinned.
 * If the node is a CPU node but passes through an input `i` directly to a pinned output `o`,
 * then the source of input `i` is also marked as pinned.
 */
void SetPinnedInputs(ExecNode *node) {
  assert(node->op != nullptr);

  // TODO(michalz): Update if/when we have passthrough for argument inputs
  int ninp = node->op->GetSpec().NumRegularInput();
  assert(static_cast<size_t>(ninp) <= node->inputs.size());

  if (node->backend != OpType::CPU) {
    for (int i = 0; i < ninp; i++) {
      auto *inp = node->inputs[i];
      inp->producer->outputs[inp->producer_output_idx].pinned = true;
    }
  } else if (node->op->GetSpec().GetSchema().HasPassThrough()) {
    auto &schema = node->op->GetSpec().GetSchema();
    int nout = node->outputs.size();
    for (int i = 0; i < ninp; i++) {
      auto *input = node->inputs[i];
      if (input->device != StorageDevice::CPU)  // we're not interested in non-CPU buffers
        continue;

      auto &source_output = input->producer->outputs[input->producer_output_idx];
      if (source_output.pinned)  // already pinned
        continue;

      for (int o = 0; o < nout; o++) {
        // If input `i` passes to a pinned output `o`, then the input should also be marked
        // as pinned. This will be followed in reverse topological order.
        if (node->outputs[o].pinned && schema.IsPassThrough(i, o, false)) {
          source_output.pinned = true;
          break;
        }
      }
    }
  }
}

}  // namespace

void ExecGraph::FindPinnedBuffers() {
  // No non-cpu ops? Just mark everything as non-pinned and we're done.
  auto is_gpu_edge = [](const ExecEdge &e) { return e.device == StorageDevice::GPU; };
  bool has_gpu_buffers = std::find_if(edges_.begin(), edges_.end(), is_gpu_edge) != edges_.end();
  if (!has_gpu_buffers) {
    for (auto &n : nodes_)
      for (auto &o : n.outputs)
        o.pinned = false;
    return;
  }

  // go in reverse topological order, from outputs to inputs
  for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
    ExecNode &n = *it;
    if (n.is_pipeline_output)
      continue;
    SetPinnedInputs(&n);
  }
}

}  // namespace exec2
}  // namespace dali
