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

#ifndef DALI_PIPELINE_EXECUTOR_EXECUTOR2_WORKSPACE_CACHE_H_
#define DALI_PIPELINE_EXECUTOR_EXECUTOR2_WORKSPACE_CACHE_H_

#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <utility>
#include <vector>
#include "dali/core/cuda_event_pool.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {
namespace exec2 {

struct ExecEnv {
  ThreadPool *thread_pool = nullptr;
  AccessOrder order = AccessOrder::host();
};

struct WorkspaceParams {
  ExecEnv *env = nullptr;
  std::shared_ptr<IterationData> iter_data;
  std::optional<int> batch_size = 0;  // TODO(michalz): add more batch size logic
};

inline void ApplyWorkspaceParams(Workspace &ws, const WorkspaceParams &params) {
  if (params.env) {
    ws.SetThreadPool(params.env->thread_pool);
    ws.set_output_order(params.env->order);
  }
  ws.InjectIterationData(params.iter_data);
  if (params.batch_size.has_value())
    ws.SetBatchSizes(*params.batch_size);
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

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR2_WORKSPACE_CACHE_H_
