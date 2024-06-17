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
#include "dali/core/nvtx.h"
#include "dali/pipeline/executor/source_info_propagation.h"

namespace dali {
namespace exec2 {

tasking::SharedTask OpTaskFunc::CreateTask(ExecNode *node, CachedWorkspace ws) {
  if (node->is_pipeline_output) {
    return tasking::Task::Create(
      OpTaskFunc(node, std::move(ws)).GetOutputTaskRunnable());
  } else {
    int nout = ws->NumOutput();
    return tasking::Task::Create(
      nout,
      OpTaskFunc(node, std::move(ws)).GetOpTaskRunnable());
  }
}

OpTaskFunc::OpTaskOutputs OpTaskFunc::Run() {
  SetWorkspaceInputs();
  SetupOp();
  RunOp();
  auto &&ret = GetWorkspaceOutputs();
  node_->PutWorkspace(std::move(ws_));
  return ret;
}


Workspace OpTaskFunc::GetOutput() {
  assert(ws_->NumInput() == 0);
  assert(ws_->NumArgumentInput() == 0);
  std::unordered_set<cudaEvent_t> events(ws_->NumOutput());

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

  if (ws_->has_event() && ws_->has_stream())
    CUDA_CALL(cudaEventRecord(ws_->event() , ws_->stream()));

  Workspace ret = *ws_;
  node_->PutWorkspace(std::move(ws_));
  return ret;
}

void OpTaskFunc::SetupOp() {
  auto &ws = *ws_;
  int nout = node_->op->GetSpec().NumOutput();
  std::vector<OutputDesc> output_descs;
  assert(ws.NumOutput() == nout);
  output_descs.resize(nout);
  // Run the operator setup
  bool should_resize;
  {
    DomainTimeRange tr("[DALI][OpTaskFunc] Setup " + GetOpDisplayName(node_->op->GetSpec()));
    should_resize = node_->op->Setup(output_descs, ws);
  }
  // If Setup returns true, we must resize the outputs;
  // otherwise we just get empty TensorLists with an expected number of samples.
  for (int i = 0; i < nout; i++) {
    if (ws.OutputIsType<CPUBackend>(i)) {
      if (!ws.OutputPtr<CPUBackend>(i)) {
        auto tl = std::make_shared<TensorList<CPUBackend>>(output_descs[i].shape.num_samples());
        ws.SetOutput(i, tl);
      }
      if (should_resize)
        ws.Output<CPUBackend>(i).Resize(output_descs[i].shape, output_descs[i].type);
    } else if (ws.OutputIsType<GPUBackend>(i)) {
      if (!ws.OutputPtr<GPUBackend>(i)) {
        auto tl = std::make_shared<TensorList<GPUBackend>>(output_descs[i].shape.num_samples());
        ws.SetOutput(i, tl);
      }
      if (should_resize)
        ws.Output<GPUBackend>(i).Resize(output_descs[i].shape, output_descs[i].type);
    } else {
      assert(!"Unreachable code - unknown backend.");
    }
  }
}

void OpTaskFunc::RunOp() {
  {
    DomainTimeRange tr("[DALI][Executor] Run");
    node_->op->Run(*ws_);
  }
  PropagateSourceInfo(*ws_);
  if (ws_->has_event() && ws_->has_stream())
    CUDA_CALL(cudaEventRecord(ws_->event(), ws_->stream()));
}

void OpTaskFunc::SetWorkspaceInputs() {
  int ti = 0;
  std::unordered_set<cudaEvent_t> events(ws_->NumInput() + ws_->NumArgumentInput());
  for (int i = 0; i < ws_->NumInput(); i++, ti++) {
    if (ws_->InputIsType<CPUBackend>(i)) {
      auto inp = TaskInput<CPUBackend>(ti);
      if (inp.event)
        events.insert(inp.event);
      ws_->SetInput(i, inp.data);
    } else {
      assert(ws_->InputIsType<GPUBackend>(i));
      auto inp = TaskInput<CPUBackend>(ti);
      if (inp.event)
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

  for (auto e : events)
    ws_->output_order().wait(e);
}

OpTaskFunc::OpTaskOutputs OpTaskFunc::GetWorkspaceOutputs() {
  OpTaskOutputs ret;
  int nout = ws_->NumOutput();
  ret.reserve(nout);
  cudaEvent_t event = ws_->has_event() ? ws_->event() : nullptr;
  for (int o = 0; o < nout; o++) {
    if (ws_->OutputIsType<CPUBackend>(o)) {
      ret.push_back(OperatorIO<CPUBackend>{ws_->OutputPtr<CPUBackend>(o), event});
    } else {
      assert(ws_->OutputIsType<GPUBackend>(o));
      ret.push_back(OperatorIO<GPUBackend>{ws_->OutputPtr<GPUBackend>(o), event});
    }
  }

  return ret;
}

void ExecNode::AddDataDeps() {
  for (auto &edge : inputs) {
    assert(edge->producer->main_task);
    main_task->Subscribe(edge->producer->main_task, edge->producer_output_idx);
  }
}


}  // namespace exec2
}  // namespace dali
