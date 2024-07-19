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

class ExecGraph::Analyzer {
 public:
  void FindPinnedBuffers(ExecGraph &g) {
    // No non-cpu ops? Just mark everything as non-pinned and we're done.
    auto is_gpu_edge = [](const ExecEdge &e) { return e.device == StorageDevice::GPU; };
    bool has_gpu_buffers = std::find_if(g.edges_.begin(), g.edges_.end(), is_gpu_edge)
                           != g.edges_.end();
    if (!has_gpu_buffers) {
      for (auto &n : g.nodes_)
        for (auto &o : n.outputs)
          o.pinned = false;
      return;
    }

    // go in reverse topological order, from outputs to inputs
    for (auto it = g.nodes_.rbegin(); it != g.nodes_.rend(); ++it) {
      ExecNode &n = *it;
      if (n.is_pipeline_output)
        continue;
      SetPinnedInputs(&n);
    }
  }

  bool HasParallelConsumers(const ExecOutputDesc &out) {
    int ncons = out.consumers.size();
    // If there's just one outgoing edge from that input, we're safe.
    if (ncons <= 1)
      return false;

    // If there are multiple edges, but they point to different inputs of the same
    // consumer, then the input is effectively consumed in parallel.
    for (int i = 1; i < ncons; i++)
      if (out.consumers[i]->consumer == out.consumers[0]->consumer)
        return true;

    // Finally, let's go over all the consumers and check if they're guarded with one
    // semaphore with MaxCount() == 1. If so, then the access to the node is sequential.
    auto sem = out.consumers[0]->consumer->concurrency;
    if (!sem)
      return true;
    if (sem->MaxCount() > 1)
      return true;
    for (size_t i = 1; i < out.consumers.size(); i++)
      if (out.consumers[i]->consumer->concurrency != sem)
        return true;
    return false;
  }

  void MarkOutputsWithParallelConsumers(ExecGraph &g) {
    for (auto &n : g.nodes_) {
      for (auto &o : n.outputs)
        o.parallel_consumers = HasParallelConsumers(o);
    }
  }

  void MarkNodesWithGPUOutputs(ExecGraph &g) {
    for (auto &n : g.nodes_) {
      n.has_gpu_outputs = false;
      for (auto &o : n.outputs)
        if (o.device == StorageDevice::GPU) {
          n.has_gpu_outputs = true;
          break;
        }
    }
  }
};

void ExecGraph::Analyze() {
  if (analyzed_)
    return;
  Analyzer a;
  a.FindPinnedBuffers(*this);
  a.MarkOutputsWithParallelConsumers(*this);
  a.MarkNodesWithGPUOutputs(*this);
  analyzed_ = true;
}


}  // namespace exec2
}  // namespace dali
