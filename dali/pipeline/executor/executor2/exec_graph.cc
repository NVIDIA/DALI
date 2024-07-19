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
#include <unordered_set>
#include <utility>
#include "dali/core/cuda_event_pool.h"
#include "dali/pipeline/executor/executor2/exec_graph.h"
#include "dali/pipeline/executor/executor2/op_task.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/graph/op_graph2.h"

namespace dali {
namespace exec2 {

using tasking::Task;
using tasking::Semaphore;
using tasking::Scheduler;

void ClearWorkspacePayload(Workspace &ws) {
  auto event = ws.has_event() ? ws.event() : nullptr;
  for (int i = 0; i < ws.NumInput(); i++) {
    // TODO(michalz): Some smarter deletion management
    // If an input has multiple consumers, we should guarantee that the buffer is not destroyed
    // before all consumers are done with it. A simple way of achieving that is waiting in the
    // stream associated with the buffer for the consumer's completion event.
    // A low-hanging optimization is to skip that when the buffer's associated stream is
    // the same stream as the one the consumer executes on. Doing so will create bubbles in
    // per-operator streams.
    // Perhaps a better approach would be to move the inputs to a dedicated deleteion stream.
    // on which we would record the completion events prior to decreasing the reference count.
    if (ws.InputIsType<CPUBackend>(i)) {
      auto &inp = ws.Input<CPUBackend>(i);
      if (event && inp.order() != ws.output_order())
        inp.order().wait(event);

      ws.SetInput<CPUBackend>(i, nullptr);
    } else if (ws.InputIsType<GPUBackend>(i)) {
      auto &inp = ws.Input<GPUBackend>(i);
      if (event && inp.order() != ws.output_order())
        inp.order().wait(event);

      ws.SetInput<GPUBackend>(i, nullptr);
    }
  }

  for (int i = 0; i < ws.NumArgumentInput(); i++)
    ws.SetArgumentInput(i, nullptr);

  for (int o = 0; o < ws.NumOutput(); o++) {
    if (ws.OutputIsType<CPUBackend>(o))
      ws.SetOutput<CPUBackend>(o, nullptr);
    else if (ws.OutputIsType<GPUBackend>(o))
      ws.SetOutput<GPUBackend>(o, nullptr);
  }
  ws.InjectIterationData(nullptr);
}

ExecNode::ExecNode(std::unique_ptr<OperatorBase> op, const graph::OpNode *def)
: op(std::move(op)), instance_name(def ? def->instance_name : "") {
  if (def) {
    backend = def->op_type;
    outputs.resize(def->outputs.size());
    inputs.resize(def->inputs.size());
  }
  if (op) {
    assert(!def || def->outputs.size() == static_cast<size_t>(op->GetSpec().NumOutput()));
    assert(!def || def->inputs.size() == static_cast<size_t>(op->GetSpec().NumInput()));
    outputs.resize(std::max<size_t>(outputs.size(), op->GetSpec().NumOutput()));
    inputs.resize(std::max<size_t>(inputs.size(), op->GetSpec().NumInput()));
  }
}

void ExecNode::PutWorkspace(CachedWorkspace ws) {
  ClearWorkspacePayload(*ws);
  workspace_cache_.Put(std::move(ws));
}

void ExecNode::CreateMainTask(std::shared_ptr<IterationData> iter, const WorkspaceParams &params) {
  main_task = OpTask::CreateTask(this, GetWorkspace(iter, params));
  if (prev)
    main_task->Succeed(prev);
}

void ExecNode::CreateAuxTasks() {
  // The concurrency of the operator may be limited within a group of operators sharing
  // one concurrency semaphore.
  // Note that operator cannot run parallel to its previous iterations and we add a temporal
  // dependency between the previous and current iteration.
  if (concurrency)
    main_task->GuardWith(concurrency);

  // The output queue depth may be limited - this is guarded by a semaphore with initial count
  // equal to the queue depth.
  // The task will acquire this semaphore prior to running (potentially delaying the start).
  // The semaphore is released when all consumers of the current task's outputs are complete.
  // This means that nobody is accessing those outputs and they've been disposed of (unless
  // forwarded down the pipeline, but we're ok with that).
  if (output_queue_limit) {
    release_outputs = Task::Create([]() {});
    release_outputs->ReleaseAfterRun(output_queue_limit);
    release_outputs->Succeed(main_task);
    for (auto &output : outputs) {
      for (auto *edge : output.consumers) {
        if (edge->consumer->main_task)
          release_outputs->Succeed(edge->consumer->main_task);
      }
    }
    main_task->Succeed(output_queue_limit);
  }
}

std::optional<tasking::TaskFuture> ExecNode::Launch(Scheduler &sched) {
  if (release_outputs)
    sched.AddSilentTask(release_outputs);
  if (is_pipeline_output) {
    return sched.AddTask(main_task);
  } else {
    sched.AddSilentTask(main_task);
    return std::nullopt;
  }
}

CachedWorkspace ExecNode::CreateOutputWorkspace() {
  assert(is_pipeline_output);
  CachedWorkspace ws(new Workspace(), {});
  for (auto &e : inputs) {
    if (e->device == StorageDevice::GPU) {
      ws->AddOutput<GPUBackend>(nullptr);
    } else {
      assert(e->device == StorageDevice::CPU);
      ws->AddOutput<CPUBackend>(nullptr);
    }
  }
  return ws;
}

CachedWorkspace ExecNode::CreateOpWorkspace() {
  assert(op);
  const OpSpec &spec = op->GetSpec();
  CachedWorkspace ws(new Workspace(), {});
  for (int i = 0, ninp = inputs.size(); i < ninp; i++) {
    bool arg = spec.IsArgumentInput(i);
    bool gpu = inputs[i]->device == StorageDevice::GPU;
    if (arg) {
      ws->AddArgumentInput(spec.ArgumentInputName(i), nullptr);
    } else if (gpu) {
      ws->AddInput(std::shared_ptr<TensorList<GPUBackend>>(nullptr));
    } else {
      ws->AddInput(std::shared_ptr<TensorList<CPUBackend>>(nullptr));
    }
  }
  for (int i = 0, nout = outputs.size(); i < nout; i++) {
    bool gpu = outputs[i].device == StorageDevice::GPU;
    if (gpu) {
      ws->AddOutput(std::shared_ptr<TensorList<GPUBackend>>(nullptr));
    } else {
      ws->AddOutput(std::shared_ptr<TensorList<CPUBackend>>(nullptr));
    }
  }
  return ws;
}

/** Obtains a worskpace from a workspace cache or, if not found, creates a new one. */
CachedWorkspace ExecNode::GetWorkspace(std::shared_ptr<IterationData> iter_data,
                                       WorkspaceParams params) {
  auto ws = workspace_cache_.Get(params);
  if (!ws) {
    if (op) {
      ws = CreateOpWorkspace();
    } else {
      ws = CreateOutputWorkspace();
    }
  }
  if (!params.env)
    params.env = &env;

  if (!ws->has_event()) {
    for (int o = 0; o < ws->NumOutput(); o++) {
      if (ws->OutputIsType<GPUBackend>(o)) {
        auto event = CUDAEventPool::instance().Get();
        ws->set_event(event.release());
      }
    }
  }

  ApplyWorkspaceParams(*ws, params);
  ws->InjectIterationData(iter_data);
  return ws;
}


void ExecGraph::Validate() {
  // The checks here are extremely defensive, but they're only run once.
  auto err = [](auto &&... msg) {
    throw std::logic_error(make_string("Internal error: ", msg...));
  };

  if (validated_)
    return;
  if (nodes_.empty()) {
    if (!edges_.empty())
      err("a graph without any node has edges.");
    return;
  }
  std::unordered_set<const ExecNode *> known_nodes(nodes_.size());
  std::unordered_set<const ExecEdge *> known_edges(edges_.size());

  for (auto &n : nodes_)
    known_nodes.insert(&n);
  for (auto &e : edges_) {
    known_edges.insert(&e);
  }

  for (auto &e : edges_) {
    if (!known_nodes.count(e.producer))
      err("an edge's producer is not a known node pointer.");
    if (!known_nodes.count(e.consumer))
      err("an edge's consumer is not a known node pointer.");

    if (e.producer_output_idx >= static_cast<int>(e.producer->outputs.size()))
      err("producer output index is out of range.");
    auto &consumer_edges = e.producer->outputs[e.producer_output_idx].consumers;
    if (std::count(consumer_edges.begin(), consumer_edges.end(), &e) != 1)
      err("the relevant producer's output doesn't have this edge as one of the consumers.");

    if (e.consumer->inputs[e.consumer_input_idx] != &e)
      err("inconsistent edge consumer vs consumer node's input.");
  }

  for (auto &n : nodes_) {
    if (n.op) {
      auto &spec = n.op->GetSpec();
      if (n.inputs.size() != static_cast<size_t>(spec.NumInput()))
        err("a node has a different number of inputs than used in the OpSpec");
      if (n.outputs.size() != static_cast<size_t>(spec.NumOutput()))
        err("a node has a different number of outputs than used in the OpSpec");
    }

    for (int o = 0, nout = n.outputs.size(); o < nout; o++) {
      auto &consumers = n.outputs[o].consumers;
      for (auto &e : consumers) {
        if (!known_edges.count(e))
          err("a node's output is not a known edge pointer.");
        if (e->producer != &n)
          err("a node's output's producer should always point to self.");
        if (e->producer_output_idx != o)
          err("a node's output's index must match its position in the output array.");
      }
    }
    for (int i = 0, ninp = n.inputs.size(); i < ninp; i++) {
      auto *e = n.inputs[i];
      if (!known_edges.count(e))
        err("a node's output is not a known edge pointer.");
      if (e->consumer != &n)
        err("a node's input's consumer should always point to self.");
      if (e->consumer_input_idx != i)
        err("a node's input index must match its position in the input array.");
    }

    bool is_last = &n == &nodes_.back();
    if (is_last != n.is_pipeline_output)
      err("there must be exactly one output node and it must be the last node in the graph.");
  }

  validated_ = true;
}

void ExecGraph::PrepareIteration(
    const std::shared_ptr<IterationData> &iter_data,
    const WorkspaceParams &params) {
  Validate();
  Analyze();
  for (auto &n : nodes_) {
    n.NextIter();
    n.CreateMainTask(iter_data, params);
  }
  for (auto &n : nodes_) {
    n.AddDataDeps();
    n.CreateAuxTasks();
  }
}

tasking::TaskFuture ExecGraph::Launch(tasking::Scheduler &sched) {
  Validate();
  Analyze();
  std::optional<tasking::TaskFuture> ret;
  for (auto &n : nodes_) {
    auto maybe_future = n.Launch(sched);
    if (maybe_future) {
      assert(!ret && "Internal error - multiple output nodes present");
      ret = std::move(maybe_future);
    }
  }
  assert(ret && "Internal error - no output node present");
  // In case of error, if the above assert is absent (in release), the following line will throw.
  return std::move(ret).value();
}

}  // namespace exec2
}  // namespace dali
