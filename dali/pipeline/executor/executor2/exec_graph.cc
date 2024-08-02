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
#include "dali/pipeline/graph/op_graph2.h"
#include "dali/pipeline/operator/batch_size_provider.h"

namespace dali {
namespace exec2 {

using tasking::Task;
using tasking::Semaphore;
using tasking::Scheduler;

void ClearWorkspacePayload(Workspace &ws) {
  auto event = ws.has_event() ? ws.event() : nullptr;
  ws.InjectIterationData(nullptr);
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

inline OpType BackendFromOp(const OperatorBase *op) {
  if (dynamic_cast<const Operator<CPUBackend>*>(op)) {
    return OpType::CPU;
  } else if (dynamic_cast<const Operator<GPUBackend>*>(op)) {
    return OpType::GPU;
  } else {
    assert(dynamic_cast<const Operator<MixedBackend>*>(op));
    return OpType::MIXED;
  }
}

ExecNode::ExecNode(std::unique_ptr<OperatorBase> op, const graph::OpNode *def)
: op(std::move(op))
, instance_name(def ? def->instance_name : "")
, backend(def ? def->op_type : op ? BackendFromOp(op.get()) : OpType::CPU)
, is_batch_size_provider(dynamic_cast<BatchSizeProvider *>(this->op.get()) != nullptr) {
  if (def) {
    outputs.resize(def->outputs.size());
    inputs.resize(def->inputs.size());
  }
  if (this->op) {
    auto &spec = this->op->GetSpec();
    assert(!def || def->outputs.size() == static_cast<size_t>(spec.NumOutput()));
    assert(!def || def->inputs.size() == static_cast<size_t>(spec.NumInput()));
    outputs.resize(std::max<size_t>(outputs.size(), spec.NumOutput()));
    inputs.resize(std::max<size_t>(inputs.size(), spec.NumInput()));
  }
}

void ExecNode::PutWorkspace(CachedWorkspace ws) {
  ClearWorkspacePayload(*ws);
  workspace_cache_.Put(std::move(ws));
}

void ExecNode::CreateMainTask(std::shared_ptr<IterationData> iter, const WorkspaceParams &params) {
  main_task_ = OpTask::CreateTask(this, GetWorkspace(iter, params));
  if (prev_task_)
    main_task_->Succeed(prev_task_);
}

void ExecNode::CreateAuxTasks() {
  // The concurrency of the operator may be limited within a group of operators sharing
  // one concurrency semaphore.
  // Note that operator cannot run parallel to its previous iterations and we add a temporal
  // dependency between the previous and current iteration.
  if (concurrency)
    main_task_->GuardWith(concurrency);

  // The output queue depth may be limited - this is guarded by a semaphore with initial count
  // equal to the queue depth.
  // The task will acquire this semaphore prior to running (potentially delaying the start).
  // The semaphore is released when all consumers of the current task's outputs are complete.
  // This means that nobody is accessing those outputs and they've been disposed of (unless
  // forwarded down the pipeline, but we're ok with that).
  if (output_queue_limit) {
    release_outputs_ = Task::Create([]() {});
    release_outputs_->ReleaseAfterRun(output_queue_limit);
    release_outputs_->Succeed(main_task_);
    for (auto &output : outputs) {
      for (auto *edge : output.consumers) {
        if (edge->consumer->main_task_)
          release_outputs_->Succeed(edge->consumer->main_task_);
      }
    }
    main_task_->Succeed(output_queue_limit);
  }
}

std::optional<tasking::TaskFuture> ExecNode::Launch(Scheduler &sched) {
  if (release_outputs_)
    sched.AddSilentTask(release_outputs_);
  if (is_pipeline_output) {
    return sched.AddTask(main_task_);
  } else {
    sched.AddSilentTask(main_task_);
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
  ws->SetOperatorInstanceName(instance_name);
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

void ExecGraph::PrepareIteration(
    const std::shared_ptr<IterationData> &iter_data,
    const WorkspaceParams &params) {
  if (nodes_.empty())
    throw std::logic_error("Cannot execute an empty graph");

  if (params.batch_size <= 0)
    throw std::runtime_error("Batch size must be a positive number.");

  Sort();
  Validate();
  Analyze();

  // Create a special task that checks the predicted batch size
  // and populates the respective field in IterationData
  auto prev_infer = infer_batch_size_task_;
  infer_batch_size_task_ = InferBatchSize(iter_data, params.batch_size);
  if (infer_batch_size_task_) {
    if (prev_infer)
      infer_batch_size_task_->Succeed(prev_infer);
  } else {
    iter_data->default_batch_size = params.batch_size;
  }

  for (auto &n : nodes_) {
    n.NextIter();
    n.CreateMainTask(iter_data, params);
  }
  for (auto &n : nodes_) {
    n.AddDataDeps();
    n.CreateAuxTasks();
    // all root tasks must succeed batch size inference
    if (n.inputs.empty() && infer_batch_size_task_)
      n.main_task_->Subscribe(infer_batch_size_task_);
  }
}

inline std::string_view NodeName(const ExecNode &n) {
  return !n.instance_name.empty() ? n.instance_name : n.op->GetSpec().SchemaName();
}

tasking::SharedTask ExecGraph::InferBatchSize(const std::shared_ptr<IterationData> &iter_data,
                                              int max_batch_size) {
  std::vector<std::pair<ExecNode *, BatchSizeProvider *>> bsps;
  for (auto &n : nodes_) {
    if (!n.is_batch_size_provider)  // these should go first
      break;
    bsps.emplace_back(&n, dynamic_cast<BatchSizeProvider *>(n.op.get()));
    assert(bsps.back().second);
  }
  if (bsps.empty())
    return nullptr;

  return tasking::Task::Create([bsps, iter = iter_data, max_batch_size]() {
    std::optional<int> bs;
    assert(!bsps.empty());
    for (auto [n, bsp] : bsps) {
      int op_bs = bsp->NextBatchSize();
      if (op_bs > max_batch_size) {
        throw std::runtime_error(make_string(
          "Batch too big! The input operator \"",
          NodeName(*n),
          "\" returned a batch size ", op_bs,
          " which is larger than 'max_batch_size' for the pipeline."));
      }
      if (bs && *bs != op_bs)
        throw std::runtime_error(make_string(
          "Batch size clash! The input operator \"",
          NodeName(*n),
          "\" returned a batch size ", op_bs,
          " which is different than ", *bs,
          " retruned by \"", NodeName(*bsps[0].first), "\"."));
      bs = op_bs;
    }
    assert(bs.has_value());
    iter->default_batch_size = *bs;
    for (auto &bsp : bsps)
      bsp.second->Advance();
  });
}


tasking::TaskFuture ExecGraph::Launch(tasking::Scheduler &sched) {
  Sort();
  Validate();
  Analyze();

  if (infer_batch_size_task_)
    sched.AddSilentTask(infer_batch_size_task_);

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
