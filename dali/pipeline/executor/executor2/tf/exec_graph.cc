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


void SchedNode::schedule(std::shared_ptr<SchedGraph> eg, tf::Taskflow &flow) {
  main_task = flow.emplace([this]() {
    runs++;

    // Run the operation
    definition->task_fn(*ws);
    // Reset the inputs once we're done
    for (int i = 0; i < ws->NumInput(); i++) {
      if (ws->InputIsType<CPUBackend>(i))
        ws->SetOutput<CPUBackend>(i, nullptr);
      else
        ws->SetOutput<GPUBackend>(i, nullptr);
    }

    for (int i = 0, nout = outputs.size(); i < nout; i++) {
      const SchedEdge &out_edge = outputs[i];
      if (out_edge.consumer) {
        Workspace &consumer_ws = out_edge.consumer->ws;
        auto &buf = ws.out[out_edge.producer_output_idx];
        if (buf)
          buf->readers++;
        consumer_ws.in[out_edge.consumer_input_idx] = buf;
      }
    }

    for (int i = 0; i < ws->NumOutput(); i++) {
      if (ws->OutputIsType<CPUBackend>(i))
        ws->SetOutput<CPUBackend>(i, nullptr);
      else
        ws->SetOutput<GPUBackend>(i, nullptr);
    }
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
    if (!in.producer->release_output.empty())
      main_task.precede(in.producer->release_output);
  }
  main_task.acquire(definition->concurrency).release(definition->concurrency);
  auto &output_queue = definition->output_queue;
  if (output_queue.has_value()) {
    main_task.acquire(*output_queue);
    release_output = flow.emplace([]() {}).release(*output_queue);
  }
}


void assert_valid(ExecGraph &eg) {
  for (auto &exec_node : eg.nodes) {
    for (int i = 0; i < exec_node.outputs.size(); i++) {
      auto *e = exec_node.outputs[i];
      assert(e->producer == &exec_node);
      assert(e->consumer->inputs[e->consumer_input_idx] == e);
    }
    for (int i = 0; i < exec_node.inputs.size(); i++) {
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
      }
    }
  }
}

void assert_valid(SchedGraph &eg) {
  for (auto &sched_node : eg.nodes) {
    assert(sched_node.definition);
    auto &exec_node = *sched_node.definition;

    auto validate_edges = [](auto &sched_edges, auto &exec_edges) {
      assert(sched_edges.size() == exec_edges.size());
      for (int i = 0; i < sched_edges.size(); i++) {
        assert(sched_edges[i].producer->definition == exec_edges[i]->producer);
        assert(sched_edges[i].consumer->definition == exec_edges[i]->consumer);
        assert(sched_edges[i].producer_output_idx == exec_edges[i]->producer_output_idx);
        assert(sched_edges[i].consumer_input_idx == exec_edges[i]->consumer_input_idx);
      }
    };

    validate_edges(sched_node.inputs, exec_node.inputs);
    validate_edges(sched_node.outputs, exec_node.outputs);

    for (int i = 0; i < sched_node.outputs.size(); i++) {
      auto &e = sched_node.outputs[i];
      assert(e.producer == &sched_node);
      assert(e.consumer->inputs[e.consumer_input_idx] == e);
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
      }
    }
  }
}


}  // namespace exec2
}  // namespace dali
