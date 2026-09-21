// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/c_api_2/pipeline_registry.h"
#include <utility>
#include "dali/core/error_handling.h"

namespace dali::c_api {

PipelineRegistry &PipelineRegistry::instance() {
  static PipelineRegistry registry;
  return registry;
}

void PipelineRegistry::Register(void *pipeline, Deleter deleter) {
  assert(pipeline && deleter);
  std::lock_guard g(mtx_);
  pipelines_[pipeline] = deleter;
}

void PipelineRegistry::Unregister(void *pipeline) {
  std::lock_guard g(mtx_);
  pipelines_.erase(pipeline);
}

bool PipelineRegistry::Destroy(void *pipeline) {
  Deleter deleter = nullptr;
  {
    std::lock_guard g(mtx_);
    auto it = pipelines_.find(pipeline);
    if (it == pipelines_.end())
      return false;
    deleter = it->second;
    pipelines_.erase(it);
  }
  deleter(pipeline);
  return true;
}

size_t PipelineRegistry::DestroyAll() {
  std::unordered_map<void *, Deleter> pipelines;
  {
    std::lock_guard g(mtx_);
    std::swap(pipelines, pipelines_);
  }
  for (auto &[pipeline, deleter] : pipelines) {
    try {
      deleter(pipeline);
    } catch (const std::exception &e) {
      DALI_WARN("Error while destroying a leaked pipeline: ", e.what());
    } catch (...) {
      DALI_WARN("Unknown error while destroying a leaked pipeline.");
    }
  }
  return pipelines.size();
}

size_t PipelineRegistry::Count() const {
  std::lock_guard g(mtx_);
  return pipelines_.size();
}

size_t GetOutstandingPipelineCount() {
  return PipelineRegistry::instance().Count();
}

size_t DestroyOutstandingPipelines() {
  return PipelineRegistry::instance().DestroyAll();
}

}  // namespace dali::c_api
