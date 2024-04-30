// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_EXECUTOR2_DYNAMIC_WORKSPACE_CACHE_H_
#define DALI_PIPELINE_EXECUTOR2_DYNAMIC_WORKSPACE_CACHE_H_

#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include "dali/core/cuda_event_pool.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {
namespace exec2 {


struct WorkspaceParams {
  ThreadPool  *thread_pool = nullptr;
  AccessOrder  order = AccessOrder::host();
  std::optional<int> batch_size = 0;  // TODO(michalz): add more batch size logic
};

inline void ApplyWorkspaceParams(Workspace &ws, const WorkspaceParams &params) {
  ws.SetThreadPool(params.thread_pool);
  ws.set_output_order(params.order);
  if (params.batch_size.has_value())
    ws.SetBatchSizes(*params.batch_size);
}

inline WorkspaceParams GetWorkspaceParams(const Workspace &ws) {
  WorkspaceParams params = {};
  params.thread_pool = ws.HasThreadPool() ? &ws.GetThreadPool() : nullptr;
  params.order = ws.output_order();
  if (ws.NumOutput())
    params.batch_size = ws.GetRequestedBatchSize(0);
  else if (ws.NumInput())
    params.batch_size = ws.GetInputBatchSize(0);
  return params;
}

struct CachedWorkspaceDeleter {
  void operator()(Workspace *ws) {
    if (ws->has_event()) {
      CUDAEvent event(ws->event());
      ws->set_event(nullptr);
      CUDAEventPool::instance().Put(std::move(event));
    }
    delete ws;
  }
};

using CachedWorkspace = std::unique_ptr<Workspace, CachedWorkspaceDeleter>;

class WorkspaceCache {
 public:
  CachedWorkspace Get(const WorkspaceParams &params) {
    CachedWorkspace ret;
    {
      std::lock_guard g(mtx_);
      if (workspaces_.empty())
        return nullptr;

      ret = std::move(workspaces_.front());
      workspaces_.pop();
    }
    ApplyWorkspaceParams(*ret, params);
    return ret;
  }

  void Put(CachedWorkspace ws) {
    std::lock_guard g(mtx_);
    workspaces_.push(std::move(ws));
  }

 private:
  std::mutex mtx_;
  std::queue<CachedWorkspace> workspaces_;

};

}  // namespace exec2
}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR2_DYNAMIC_WORKSPACE_CACHE_H_