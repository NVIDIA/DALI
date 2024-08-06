// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>
#include "dali/pipeline/executor/executor2/op_task.h"
#include "dali/pipeline/executor/executor2/exec_graph.h"
#include "dali/pipeline/executor/source_info_propagation.h"
#include "dali/core/nvtx.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/checkpointing/checkpoint.h"
#include "dali/core/call_at_exit.h"
#include "dali/pipeline/operator/error_reporting.h"

namespace dali {
namespace exec2 {

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
      if (auto &pinp = ws.InputPtr<CPUBackend>(i)) {
        auto &inp = *pinp;
        if (event && inp.order() != ws.output_order())
          inp.order().wait(event);
        ws.SetInput<CPUBackend>(i, nullptr);
      }
    } else if (ws.InputIsType<GPUBackend>(i)) {
      if (auto &pinp = ws.InputPtr<GPUBackend>(i)) {
          auto &inp = *pinp;
        if (event && inp.order() != ws.output_order())
          inp.order().wait(event);
        ws.SetInput<GPUBackend>(i, nullptr);
      }
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

void OpTask::ClearWorkspace() {
  assert(ws_);
  ClearWorkspacePayload(*ws_);
}

tasking::SharedTask OpTask::CreateTask(ExecNode *node, const WorkspaceParams &params) {
  if (node->is_pipeline_output) {
    return tasking::Task::Create(
      OpTask(node, params).GetOutputTaskRunnable());
  } else {
    int nout = node->outputs.size();
    return tasking::Task::Create(
      nout,
      OpTask(node, params).GetOpTaskRunnable());
  }
}

OpTask::OpTaskOutputs OpTask::Run() {
  // SetWorkspaceInputs must not go into the try/catch because it rethrows errors
  // from the inputs and we don't want them to be wrapped again as this operator's error.
  auto workspace_scope = GetWorkspace();
  SetWorkspaceInputs();
  try {
    SetupOp();
    RunOp();
    auto &&ret = GetWorkspaceOutputs();
    return ret;
  } catch (...) {
    PropagateError({
      std::current_exception(),
      "Critical error in pipeline:\n" + GetErrorContextMessage(node_->op->GetSpec()),
      "\nCurrent pipeline object is no longer valid."});
  }
}


PipelineOutput OpTask::GetOutput() {
  auto workspace_scope = GetWorkspace();
  assert(ws_->NumInput() == 0);
  assert(ws_->NumArgumentInput() == 0);
  std::unordered_set<cudaEvent_t> events(ws_->NumOutput());
  assert(ws_->NumOutput() == static_cast<int>(node_->inputs.size()));

  for (int o = 0; o < ws_->NumOutput(); o++) {
    if (ws_->OutputIsType<CPUBackend>(o)) {
      auto &inp = TaskInput<CPUBackend>(o);
      if (inp.event)
        events.insert(inp.event);
      ws_->SetOutput(o, inp.data);
    } else {
      assert(ws_->OutputIsType<GPUBackend>(o));
      auto &inp = TaskInput<GPUBackend>(o);
      if (inp.event)
        events.insert(inp.event);
      ws_->SetOutput(o, inp.data);
    }
  }

  for (auto e : events)
    ws_->output_order().wait(e);

  for (int o = 0; o < ws_->NumOutput(); o++) {
    if (ws_->OutputIsType<CPUBackend>(o)) {
      auto &out = ws_->Output<CPUBackend>(o);
      out.set_order(ws_->output_order(), false);
    } else {
      auto &out = ws_->Output<GPUBackend>(o);
      out.set_order(ws_->output_order(), false);
    }
  }

  std::optional<int> device = {};
  if (ws_->output_order().is_device()) {
    assert(ws_->event());
    device = ws_->output_order().device_id();
    CUDA_CALL(cudaEventRecord(ws_->event(), ws_->output_order().stream()));
  }

  PipelineOutput ret{ *ws_, event_, device };
  return ret;
}

bool OpTask::ShouldSkip() const {
  int ninp_reg = ws_->NumInput();
  int ninp_arg = ws_->NumArgumentInput();
  if (ninp_reg || ninp_arg) {
    for (int i = 0; i < ninp_reg; i++) {
      if (ws_->GetInputBatchSize(i) != 0)
        return false;
    }
    for (int i = 0; i < ninp_arg; i++) {
      if (ws_->ArgumentInput(i).num_samples() != 0)
        return false;
    }
    return true;
  } else {
    for (int i = 0; i < ws_->NumOutput(); i++)
      if (ws_->GetRequestedBatchSize(0) != 0)
        return false;

    return true;
  }
}

template <typename Backend>
void OpTask::ApplyDefaultLayout(int input_idx, const OpSchema &schema) {
  TensorList<Backend> &input = ws_->UnsafeMutableInput<Backend>(input_idx);

  TensorLayout layout_found = input.GetLayout();

  // Validate the layout_found and possibly get the default
  auto layout = schema.GetInputLayout(input_idx, input.sample_dim(), layout_found);
  if (layout == layout_found)
    return;  // no need to adjust anything
  assert(layout_found.empty() && "Layout found must match the final layout or be empty.");

  auto *input_edge = node_->inputs[input_idx];
  auto &source = input_edge->producer->outputs[input_edge->producer_output_idx];
  if (!source.parallel_consumers) {
    // If there's just one consumer, we can just set the layout
    input.SetLayout(layout);
    if (source.consumers.size() > 1_uz) {
      // There are multiple (sequential) consumers - we must reset the layout for the next consumer
      reset_input_layouts_.push_back(input_idx);
    }
  } else {
    // If there are multiple consumers, then we have to create a new object
    // sharing the data, as setting it in-place might cause a race condition.
    auto new_input = std::make_shared<TensorList<Backend>>();
    new_input->ShareData(input);
    new_input->SetLayout(layout);
    ws_->SetInput(input_idx, std::move(new_input));
  }
}

void OpTask::ResetInputLayouts() {
  for (int i : reset_input_layouts_) {
    if (ws_->InputIsType<CPUBackend>(i)) {
      ws_->UnsafeMutableInput<CPUBackend>(i).SetLayout({});
    } else {
      assert(ws_->InputIsType<GPUBackend>(i));
      ws_->UnsafeMutableInput<GPUBackend>(i).SetLayout({});
    }
  }
}


void OpTask::ApplyDefaultLayouts() {
  auto &schema = node_->op->GetSpec().GetSchema();
  for (int i = 0; i < ws_->NumInput(); i++) {
    if (ws_->InputIsType<CPUBackend>(i)) {
      ApplyDefaultLayout<CPUBackend>(i, schema);
    } else {
      assert(ws_->InputIsType<GPUBackend>(i));
      ApplyDefaultLayout<GPUBackend>(i, schema);
    }
  }
}

void OpTask::SetupOp() {
  auto &ws = *ws_;
  int nout = node_->outputs.size();
  assert(ws.NumOutput() == nout);

  skip_ = ShouldSkip();

  int device = -1;

  for (int i = 0; i < nout; i++) {
    if (ws.OutputIsType<CPUBackend>(i)) {
      assert(!ws.OutputPtr<CPUBackend>(i));
      auto tl = std::make_shared<TensorList<CPUBackend>>();
      bool pinned = node_->outputs[i].pinned;
      tl->set_pinned(pinned);
      if (pinned) {
        if (ws.output_order().is_device())  // Only change the order of device-ordered CPU outputs
          tl->set_order(ws.output_order());
        if (device < 0)
          CUDA_CALL(cudaGetDevice(&device));
        tl->set_device_id(device);
      }
      ws.SetOutput(i, tl);
    } else if (ws.OutputIsType<GPUBackend>(i)) {
      assert(!ws.OutputPtr<GPUBackend>(i));
      auto tl = std::make_shared<TensorList<GPUBackend>>();
      tl->set_order(ws.output_order());
      if (device < 0)
        CUDA_CALL(cudaGetDevice(&device));
      tl->set_device_id(device);
      ws.SetOutput(i, tl);
    } else {
      assert(!"Unreachable code - unknown backend.");
    }
  }

  if (!skip_) {
    DomainTimeRange tr("[DALI][OpTask] Setup " + GetOpDisplayName(node_->op->GetSpec()));
    ApplyDefaultLayouts();
    std::vector<OutputDesc> output_descs;
    output_descs.reserve(nout);
    // If Setup returns true, we must resize the outputs;
    if (node_->op->Setup(output_descs, ws)) {
      assert(output_descs.size() == static_cast<size_t>(nout));
      for (int i = 0; i < nout; i++) {
        if (ws.OutputIsType<CPUBackend>(i)) {
          ws.Output<CPUBackend>(i).Resize(output_descs[i].shape, output_descs[i].type);
        } else if (ws.OutputIsType<GPUBackend>(i)) {
          auto &output = ws.Output<GPUBackend>(i);
          output.Resize(output_descs[i].shape, output_descs[i].type);
        } else {
          assert(!"Unreachable code - unknown backend.");
        }
      }
    }
  }
}

void OpTask::RunOp() {
  if (!skip_) {
    DomainTimeRange tr("[DALI][Executor] Run");
    node_->op->Run(*ws_);
    ResetInputLayouts();
    PropagateSourceInfo(*ws_);
  }
  if (auto cpt = ws_->GetIterationData()->checkpoint) {
    node_->op->SaveState(cpt->GetOpCheckpoint(node_->instance_name), ws_->output_order());
  }
  if (ws_->has_stream()) {
    assert(ws_->has_event());
    CUDA_CALL(cudaEventRecord(ws_->event(), ws_->stream()));
  }
}

void OpTask::SetWorkspaceInputs() {
  int ti = 0;
  assert(ws_->NumInput() + ws_->NumArgumentInput() == static_cast<int>(node_->inputs.size()));
  auto order = ws_->output_order();
  std::unordered_set<cudaEvent_t> events(ws_->NumInput() + ws_->NumArgumentInput());
  for (int i = 0; i < ws_->NumInput(); i++, ti++) {
    if (ws_->InputIsType<CPUBackend>(i)) {
      auto inp = TaskInput<CPUBackend>(ti);
      if (inp.event && inp.order != order)
        events.insert(inp.event);
      ws_->SetInput(i, inp.data);
    } else {
      assert(ws_->InputIsType<GPUBackend>(i));
      auto inp = TaskInput<GPUBackend>(ti);
      // If the output order of the operator is `host` then we don't wait for GPU
      // inputs - they can't be accessed directly on host and the operator will
      // have to issue some sort of synchronization if and when necessary.
      // This optimization is essential to avoid oversynchronization
      // when the operator needs to access the metadata only (e.g. getting the shape).
      if (order.is_device() /*see comment above */ && inp.event && inp.order != order)
        events.insert(inp.event);
      ws_->SetInput(i, inp.data);
    }
  }

  for (int i = 0; i < ws_->NumArgumentInput(); i++, ti++) {
    auto &inp = TaskInput<CPUBackend>(ti);
    if (inp.event)
      events.insert(inp.event);
    ws_->SetArgumentInput(i, inp.data);
  }

  if (task_->NumInputs() > ti) {  // Artificial error input
    // TODO(michalz): Add early failure mechanism to `dali::tasking` that makes the successors
    //                of a task that threw an exception fail. Currently the error disappears,
    //                so we have to add an artificial input which propagates the exception.
    task_->GetInputValue<void>(ti++);
  }

  for (auto e : events)
    ws_->output_order().wait(e);

  if (ws_->NumOutput()) {
    if (ws_->NumInput() > 0) {
      ws_->SetBatchSizes(ws_->GetInputBatchSize(0));
    } else if (ws_->NumArgumentInput() > 0) {
      ws_->SetBatchSizes(ws_->ArgumentInput(0).num_samples());
    } else {
      ws_->SetBatchSizes(ws_->GetIterationData()->default_batch_size);
    }
  }
}

AccessOrder OpTask::OutputConsumerOrder(int output_idx) {
  assert(static_cast<size_t>(output_idx) < node_->outputs.size());
  // Return the common stream.
  auto &consumers = node_->outputs[output_idx].consumers;
  if (consumers.empty())
    return {};  // definitely no consumer
  AccessOrder order = consumers[0]->consumer->env.order;
  for (size_t i = 1; i < consumers.size(); i++)
    if (consumers[i]->consumer->env.order != order)
      return {};
  return order;
}

OpTask::OpTaskOutputs OpTask::GetWorkspaceOutputs() {
  OpTaskOutputs ret;
  int nout = ws_->NumOutput();
  ret.reserve(nout);
  auto order = ws_->output_order();
  for (int o = 0; o < nout; o++) {
    if (ws_->OutputIsType<CPUBackend>(o)) {
      auto ptr = ws_->OutputPtr<CPUBackend>(o);
      if (!ptr->shares_data()) {
        if (AccessOrder consumer_order = OutputConsumerOrder(o))
          ptr->set_order(consumer_order, false);
      }
      ret.push_back(OperatorIO<CPUBackend>{std::move(ptr), event_, order});
    } else {
      assert(ws_->OutputIsType<GPUBackend>(o));
      auto ptr = ws_->OutputPtr<GPUBackend>(o);
      if (!ptr->shares_data()) {
        if (AccessOrder consumer_order = OutputConsumerOrder(o)) {
          // Don't set the consumer order to `host`, because we don't synchronize on host.
          if (consumer_order != AccessOrder::host())
            ptr->set_order(consumer_order, false);
        }
      }
      ret.push_back(OperatorIO<GPUBackend>{ws_->OutputPtr<GPUBackend>(o), event_, order});
    }
  }

  return ret;
}

}  // namespace exec2
}  // namespace dali
