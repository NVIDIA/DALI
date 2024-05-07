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
#include <utility>
#include <vector>
#include "op_task.h"
#include "exec_graph.h"
#include "dali/core/nvtx.h"
#include "dali/pipeline/executor/source_info_propagation.h"

namespace dali {
namespace exec2 {

auto OpTaskFunc::GetOutputTaskRunnable() && {
  assert(node_->is_pipeline_output);
  return [self = std::move(*this)](tasking::Task *t) mutable {
    self.task_ = t;
    return self.GetOutput();
  };
}

auto OpTaskFunc::GetOpTaskRunnable() && {
  assert(!node_->is_pipeline_output);
  return [self = std::move(*this)](tasking::Task *t) mutable {
    self.task_ = t;
    return self.Run();
  };
}

tasking::SharedTask OpTaskFunc::CreateTask(ExecNode *node, CachedWorkspace ws) {
  if (node->is_pipeline_output) {
    return tasking::Task::Create(
      ws->NumOutput(),
      OpTaskFunc(node, std::move(ws)).GetOutputTaskRunnable());
  } else {
    return tasking::Task::Create(
      ws->NumOutput(),
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
  for (int o = 0; o < ws_->NumOutput(); o++) {
    if (ws_->OutputIsType<CPUBackend>(o)) {
      ws_->SetOutput(o, TaskInput<CPUBackend>(o));
    } else {
      assert(ws_->OutputIsType<GPUBackend>(o));
      ws_->SetOutput(o, TaskInput<GPUBackend>(o));
    }
  }

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
  for (int i = 0; i < ws_->NumInput(); i++, ti++) {
    if (ws_->InputIsType<CPUBackend>(i)) {
      ws_->SetInput(i, TaskInput<CPUBackend>(ti));
    } else {
      assert(ws_->InputIsType<GPUBackend>(i));
      ws_->SetInput(i, TaskInput<GPUBackend>(ti));
    }
  }

  for (int i = 0; i < ws_->NumArgumentInput(); i++, ti++) {
    ws_->SetArgumentInput(i, TaskInput<CPUBackend>(ti));
  }
}

OpTaskFunc::OpTaskOutputs OpTaskFunc::GetWorkspaceOutputs() {
  OpTaskOutputs ret;
  int nout = ws_->NumOutput();
  ret.reserve(nout);
  for (int o = 0; o < nout; o++) {
    if (ws_->OutputIsType<CPUBackend>(o)) {
      ret.push_back(ws_->OutputPtr<CPUBackend>(o));
    } else {
      assert(ws_->OutputIsType<CPUBackend>(o));
      ret.push_back(ws_->OutputPtr<CPUBackend>(o));
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
