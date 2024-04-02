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

#include "exec_graph.h"

namespace dali {
namespace exec2 {

template <typename Case>
auto backend_switch(StorageDevice device, Case &&callable)
{
  if (device == StorageDevice::CPU)
    return callable(CPUBackend());
  else
    return callable(GPUBackend());
}

void SchedNode::task_setup() {
  int nout = definition->op->GetSpec().NumOutput();
  std::vector<OutputDesc> output_descs;
  output_descs.resize(nout);
  if (definition->op->Setup(output_descs, *ws)) {
    for (int i = 0; i < nout; i++) {
      if (ws->OutputIsType<CPUBackend>(i)) {
        if (!ws->OutputPtr<CPUBackend>(i)) {
          auto tl = std::make_shared<TensorList<CPUBackend>>(output_descs[i].shape.num_samples());
          ws->SetOutput(i, tl);
        }
        ws->Output<CPUBackend>(i).Resize(output_descs[i].shape, output_descs[i].type);
      } else if (ws->OutputIsType<GPUBackend>(i)) {
        if (!ws->OutputPtr<GPUBackend>(i)) {
          auto tl = std::make_shared<TensorList<GPUBackend>>(output_descs[i].shape.num_samples());
          ws->SetOutput(i, tl);
        }
        ws->Output<GPUBackend>(i).Resize(output_descs[i].shape, output_descs[i].type);
      } else {
        assert(!"Unreachable code - uknonw backend.");
      }
    }
  }
}

void SchedNode::task_run() {
  definition->op->Run(*ws);
  if (ws->has_event() && ws->has_stream())
    CUDA_CALL(cudaEventRecord(ws->event(), ws->stream()));
}

void SchedNode::task_reset_inputs() {
  for (int i = 0; i < ws->NumInput(); i++) {
    if (ws->InputIsType<CPUBackend>(i))
      ws->SetInput<CPUBackend>(i, nullptr);
    else
      ws->SetInput<GPUBackend>(i, nullptr);
  }
}

void SchedNode::task_propagate_outputs() {
  for (int i = 0, nout = outputs.size(); i < nout; i++) {
    const SchedEdge &out_edge = outputs[i];
    if (out_edge.consumer) {
      Workspace &consumer_ws = *out_edge.consumer->ws;
      backend_switch(out_edge.device, [&](auto backend) {
        consumer_ws.SetInput(
          out_edge.consumer_input_idx,
          ws->OutputPtr(out_edge.producer_output_idx, backend));
      });
    }
  }
}

void SchedNode::task_reset_outputs() {
  for (int i = 0; i < ws->NumOutput(); i++) {
    if (outputs[i].pipeline_output)
      continue;  // do not reset pipeline outputs
    if (ws->OutputIsType<CPUBackend>(i))
      ws->SetOutput<CPUBackend>(i, nullptr);
    else
      ws->SetOutput<GPUBackend>(i, nullptr);
  }
}

void SchedNode::schedule(std::shared_ptr<SchedGraph> eg, tf::Taskflow &flow) {
  main_task = flow.emplace([this]() {
    // Run the operator
    task_setup();
    task_run();
    // Reset the inputs once we're done
    task_reset_inputs();
    task_propagate_outputs();
    task_reset_outputs();
  });

  for (auto &in : inputs) {
    if (!in.producer)  // external input?
      continue;
    assert(!in.producer->main_task.empty());
    // Obviously, this task succeeds any producer task
    main_task.succeed(in.producer->main_task);

    // If the preceding task has a "release_outputs" task, our current task needs to precede it -
    // that is, This way the producer will not have its output queue semaphore lowered until all
    // consumers are done.
    if (!in.producer->release_outputs.empty())
      main_task.precede(in.producer->release_outputs);
  }
  main_task.acquire(definition->concurrency).release(definition->concurrency);
  auto &output_queue = definition->output_queue;
  if (output_queue.has_value()) {
    main_task.acquire(*output_queue);
    release_outputs = flow.emplace([]() {}).release(*output_queue);
  }
}


void assert_valid(ExecGraph &eg) {
  for (auto &exec_node : eg.nodes) {
    for (int i = 0, nout = exec_node.outputs.size(); i < nout; i++) {
      auto *e = exec_node.outputs[i];
      assert(e->producer == &exec_node);
      assert(e->consumer->inputs[e->consumer_input_idx] == e);
    }
    for (int i = 0, ninp = exec_node.inputs.size(); i < ninp; i++) {
      auto *e = exec_node.inputs[i];
      assert(e->consumer == &exec_node);
      if (e->producer) {
        bool found = false;
        for (auto &out_e : e->producer->outputs) {
          if (out_e == e) {
            found = true;
            break;
          }
        }
        assert(found);
        (void)found;
      }
    }
  }
  for (auto *out : eg.outputs)
    assert(out != nullptr);
}

void assert_valid(SchedGraph &eg) {
  for (auto &e : eg.edges) {
    if (e.producer)
      assert(e.producer >= eg.nodes.data() && e.producer < eg.nodes.data() + eg.nodes.size());
    if (e.consumer)
      assert(e.consumer >= eg.nodes.data() && e.consumer < eg.nodes.data() + eg.nodes.size());
  }

  for (auto &sched_node : eg.nodes) {
    assert(sched_node.definition);
    auto &exec_node = *sched_node.definition;

    auto validate_edges = [](auto &sched_edges, auto &exec_edges) {
      assert(static_cast<size_t>(sched_edges.size()) == exec_edges.size());
      for (int i = 0; i < sched_edges.size(); i++) {
        if (sched_edges[i].producer) {
          assert(sched_edges[i].producer->definition == exec_edges[i]->producer);
          assert(sched_edges[i].producer_output_idx == exec_edges[i]->producer_output_idx);
        }
        if (sched_edges[i].consumer) {
          assert(sched_edges[i].consumer->definition == exec_edges[i]->consumer);
          assert(sched_edges[i].consumer_input_idx == exec_edges[i]->consumer_input_idx);
        }
      }
    };

    validate_edges(sched_node.inputs, exec_node.inputs);
    validate_edges(sched_node.outputs, exec_node.outputs);

    for (int i = 0; i < sched_node.outputs.size(); i++) {
      auto &e = sched_node.outputs[i];
      assert(e.producer == &sched_node);
      if (e.consumer) {
        assert(e.consumer->inputs[e.consumer_input_idx] == e);
      }
    }
    for (int i = 0; i < sched_node.inputs.size(); i++) {
      auto &e = sched_node.inputs[i];
      assert(e.consumer == &sched_node);
      if (e.producer) {
        bool found = false;
        for (auto &out_e : e.producer->outputs) {
          if (out_e == e) {
            found = true;
            break;
          }
        }
        assert(found);
        (void)found;
      }
    }
  }
  for (auto *out : eg.outputs)
    assert(out != nullptr);
}

inline ptrdiff_t ptrdiff(const void *a, const void *b) {
  return reinterpret_cast<intptr_t>(a) - reinterpret_cast<intptr_t>(b);
}

template <typename T>
inline T *ptradd(T *a, ptrdiff_t byte_offset) {
  return reinterpret_cast<T*>(reinterpret_cast<intptr_t>(a) + byte_offset);
}

void SchedGraph::init(ExecGraph &def, const WorkspaceParams &params) {
  std::unordered_map<ExecNode *, int> node_indices(def.nodes.size());
  nodes.clear();
  edges.clear();

  std::vector<ExecNode *> sorted;
  def.sort_and_prune(sorted);

  int num_edges = 0;
  for (auto *node : sorted) {
    node_indices.insert({node, node_indices.size()});
    num_edges += node->inputs.size() + node->outputs.size();
  }

  edges.resize(num_edges);
  nodes.resize(sorted.size());
  outputs.resize(def.outputs.size());
  std::unordered_map<const ExecEdge *, int> output_idx(outputs.size());
  for (int i = 0, n = def.outputs.size(); i < n; i++)
    output_idx[def.outputs[i]] = i;

  int i = 0;
  int e = 0;
  for (auto &exec_node : sorted) {
    auto &sched_node = nodes[i++];
    sched_node.definition = exec_node;
    sched_node.ws = sched_node.definition->GetWorkspace(params);
    SchedEdge *inp = &edges[e];
    for (auto *exec_edge : exec_node->inputs) {
      assert(e < (int)edges.size());
      auto &sched_edge = edges[e++];
      sched_edge.producer_output_idx = exec_edge->producer_output_idx;
      sched_edge.consumer_input_idx = exec_edge->consumer_input_idx;
      if (exec_edge->producer)
        sched_edge.producer = &nodes[node_indices[exec_edge->producer]];
      if (exec_edge->consumer)
        sched_edge.consumer = &nodes[node_indices[exec_edge->consumer]];
    }
    SchedEdge *out = &edges[e];
    for (auto *exec_edge : exec_node->outputs) {
      assert(e < (int)edges.size());
      auto &sched_edge = edges[e++];
      sched_edge.producer_output_idx = exec_edge->producer_output_idx;
      sched_edge.consumer_input_idx = exec_edge->consumer_input_idx;
      if (exec_edge->producer)
        sched_edge.producer = &nodes[node_indices[exec_edge->producer]];
      if (exec_edge->consumer)
        sched_edge.consumer = &nodes[node_indices[exec_edge->consumer]];
      auto it = output_idx.find(exec_edge);
      if (it != output_idx.end()) {
        out->pipeline_output = true;
        outputs[it->second] = out;
      }
    }
    SchedEdge *end = &edges[e];
    sched_node.inputs = span(inp, out);
    sched_node.outputs = span(out, end);
  }
  assert(static_cast<size_t>(e) == edges.size());
  assert_valid(*this);
}

SchedGraph &SchedGraph::operator=(const SchedGraph &g) {
  nodes.resize(g.nodes.size());
  edges = g.edges;
  outputs = g.outputs;

  int V = nodes.size();
  int E = edges.size();

  ptrdiff_t edge_fixup = ptrdiff(edges.data(), g.edges.data());
  ptrdiff_t node_fixup = ptrdiff(nodes.data(), g.nodes.data());

  for (auto &e : edges) {
    if (e.producer) {
      assert(e.producer >= &g.nodes.front() && e.producer <= &g.nodes.back());
      e.producer = ptradd(e.producer, node_fixup);
      assert(e.producer >= &nodes.front() && e.producer <= &nodes.back());
    }
    if (e.consumer) {
      assert(e.consumer >= &g.nodes.front() && e.consumer <= &g.nodes.back());
      e.consumer = ptradd(e.consumer, node_fixup);
      assert(e.consumer >= &nodes.front() && e.consumer <= &nodes.back());
    }
  }
  for (auto *&e : outputs) {
    e = ptradd(e, edge_fixup);
    assert(e >= &edges.front() && e <= &edges.back());
  }

  for (int i = 0; i < V; i++) {
    nodes[i].definition = g.nodes[i].definition;
    nodes[i].inputs = nodes[i].outputs = {};
    nodes[i].ws = nodes[i].definition->GetWorkspace(GetWorkspaceParams(*g.nodes[i].ws));

    if (!g.nodes[i].inputs.empty())
      nodes[i].inputs =
          span(ptradd(g.nodes[i].inputs.data(), edge_fixup), g.nodes[i].inputs.size());
    if (!g.nodes[i].outputs.empty())
      nodes[i].outputs =
          span(ptradd(g.nodes[i].outputs.data(), edge_fixup), g.nodes[i].outputs.size());

    assert(nodes[i].inputs.data() == nullptr ||
            (nodes[i].inputs.data() >= edges.data() &&
            nodes[i].inputs.data() + nodes[i].inputs.size() <= edges.data() + edges.size()));

    assert(nodes[i].outputs.data() == nullptr ||
            (nodes[i].outputs.data() >= edges.data() &&
            nodes[i].outputs.data() + nodes[i].outputs.size() <= edges.data() + edges.size()));
  }

  assert_valid(*this);
  return *this;
}

}  // namespace exec2
}  // namespace dali
