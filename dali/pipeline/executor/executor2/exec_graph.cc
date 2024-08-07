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
#include "dali/pipeline/executor/executor2/exec_node_task.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/graph/op_graph2.h"
#include "dali/pipeline/operator/batch_size_provider.h"
#include "dali/pipeline/operator/error_reporting.h"

namespace dali {
namespace exec2 {

using tasking::Task;
using tasking::Scheduler;

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
, backend(def ? def->op_type : this->op ? BackendFromOp(this->op.get()) : OpType::CPU)
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

void ExecNode::CreateMainTask(const WorkspaceParams &params) {
  main_task_ = ExecNodeTask::CreateTask(this, params);
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

std::unique_ptr<Workspace> ExecNode::CreateOutputWorkspace() {
  assert(is_pipeline_output);
  auto ws = std::make_unique<Workspace>();
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

std::unique_ptr<Workspace> ExecNode::CreateOpWorkspace() {
  assert(op);
  const OpSpec &spec = op->GetSpec();
  auto ws = std::make_unique<Workspace>();
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
std::pair<std::unique_ptr<Workspace>, SharedEventLease>
ExecNode::GetWorkspace(WorkspaceParams params) {
  if (!ws_) {
    assert(!has_workspace_);
    if (op) {
      ws_ = CreateOpWorkspace();
    } else {
      ws_ = CreateOutputWorkspace();
    }
    has_workspace_ = true;
  }
  if (!params.env)
    params.env = &env;

  ws_event_.reset();
  ws_->set_event(nullptr);

  for (int o = 0; o < ws_->NumOutput(); o++) {
    if (ws_->OutputIsType<GPUBackend>(o)) {
      if (!ws_event_)
        ws_event_ = SharedEventLease::Get();
      ws_->set_event(ws_event_);
      break;
    }
  }

  ApplyWorkspaceParams(*ws_, params);
  return { std::move(ws_), ws_event_ };
}

void ExecNode::PutWorkspace(std::unique_ptr<Workspace> &&ws) {
  assert(has_workspace_);
  assert(ws);
  assert(!ws_);
  assert(!ws->has_event() || ws->event() == ws_event_.get());
  ws_ = std::move(ws);
}

void ExecNode::AddDataDeps() {
  for (auto &edge : inputs) {
    assert(edge->producer->main_task_);
    main_task_->Subscribe(edge->producer->main_task_, edge->producer_output_idx);
  }
}

void ExecGraph::PrepareIteration(const WorkspaceParams &params) {
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
  infer_batch_size_task_ = InferBatchSize(params.iter_data, params.batch_size);
  if (infer_batch_size_task_) {
    if (prev_infer)
      infer_batch_size_task_->Succeed(prev_infer);
  } else {
    params.iter_data->default_batch_size = params.batch_size;
  }

  for (auto &n : nodes_) {
    n.NextIter();
    n.CreateMainTask(params);
  }
  for (auto &n : nodes_) {
    n.AddDataDeps();
    n.CreateAuxTasks();
    // all root tasks must succeed batch size inference
    if (n.inputs.empty() && infer_batch_size_task_)
      n.main_task_->Subscribe(infer_batch_size_task_);
  }

  iteration_prepared_ = true;
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

  return Task::Create([bsps, iter = iter_data, max_batch_size]() {
    std::optional<int> bs;
    assert(!bsps.empty());
    for (auto [n, bsp] : bsps) {
      int op_bs = bsp->NextBatchSize();
      try {
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
      } catch (...) {
        PropagateError({
          std::current_exception(),
          "Critical error in pipeline:\n" + GetErrorContextMessage(n->op->GetSpec()),
          "\nCurrent pipeline object is no longer valid."});
      }
    }
    assert(bs.has_value());
    iter->default_batch_size = *bs;
    for (auto [n, bsp] : bsps) {
      try {
        bsp->Advance();
      } catch (...) {
        PropagateError({
          std::current_exception(),
          "Critical error in pipeline:\n" + GetErrorContextMessage(n->op->GetSpec()),
          "\nCurrent pipeline object is no longer valid."});
      }
    }
  });
}


tasking::TaskFuture ExecGraph::Launch(tasking::Scheduler &sched) {
  if (!iteration_prepared_)
    throw std::logic_error("Launch called before PrepareIteration");
  iteration_prepared_ = false;
  assert(sorted_ && analyzed_ && validated_);

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
