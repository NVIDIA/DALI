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
#include "dali/core/cuda_event_pool.h"
#include "dali/pipeline/executor/executor2/exec_graph.h"
#include "dali/pipeline/executor/executor2/op_task.h"
#include "dali/pipeline/operator/op_spec.h"

namespace dali {
namespace exec2 {

using tasking::Task;
using tasking::Semaphore;
using tasking::Scheduler;

void ClearWorkspacePayload(Workspace &ws) {
  for (int i = 0; i < ws.NumInput(); i++) {
    if (ws.InputIsType<CPUBackend>(i))
      ws.SetInput<CPUBackend>(i, nullptr);
    else if (ws.InputIsType<GPUBackend>(i))
      ws.SetInput<GPUBackend>(i, nullptr);
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
    for (auto &edge : outputs) {
      if (edge->consumer->main_task)
        release_outputs->Succeed(edge->consumer->main_task);
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
  bool has_gpu_outputs = false;
  for (auto &e : inputs) {
    if (e->device == StorageDevice::GPU) {
      ws->AddOutput<GPUBackend>(nullptr);
      has_gpu_outputs = true;
    } else {
      assert(e->device == StorageDevice::CPU);
      ws->AddOutput<CPUBackend>(nullptr);
    }
  }
  if (has_gpu_outputs) {
    auto event = CUDAEventPool::instance().Get();
    ws->set_event(event.release());
  }
  return ws;
}

CachedWorkspace ExecNode::CreateOpWorkspace() {
  assert(op);
  const OpSpec &spec = op->GetSpec();
  CachedWorkspace ws(new Workspace(), {});
  bool has_gpu_outputs = false;
  for (int i = 0; i < spec.NumInput(); i++) {
    bool arg = spec.IsArgumentInput(i);
    bool gpu = spec.InputDevice(i) == "gpu";
    has_gpu_outputs |= gpu;
    if (arg) {
      ws->AddArgumentInput(spec.ArgumentInputName(i), nullptr);
    } else if (gpu) {
      ws->AddInput(std::shared_ptr<TensorList<GPUBackend>>(nullptr));
    } else {
      ws->AddInput(std::shared_ptr<TensorList<CPUBackend>>(nullptr));
    }
  }
  for (int i = 0; i < spec.NumOutput(); i++) {
    bool gpu = spec.OutputDevice(i) == "gpu";
    if (gpu) {
      has_gpu_outputs = true;
      ws->AddOutput(std::shared_ptr<TensorList<GPUBackend>>(nullptr));
    } else {
      ws->AddOutput(std::shared_ptr<TensorList<CPUBackend>>(nullptr));
    }
  }
  if (has_gpu_outputs) {
    CUDAEvent event = CUDAEventPool::instance().Get();
    ws->set_event(event.release());
  }
  return ws;
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
}

void ExecGraph::PrepareIteration(
    const std::shared_ptr<IterationData> &iter_data,
    const WorkspaceParams &params) {
  for (auto &n : nodes) {
    n.NextIter();
    n.CreateMainTask(iter_data, params);
  }
  for (auto &n : nodes) {
    n.AddDataDeps();
    n.CreateAuxTasks();
  }
}

tasking::TaskFuture ExecGraph::Launch(tasking::Scheduler &sched) {
  std::optional<tasking::TaskFuture> ret;
  for (auto &n : nodes) {
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
