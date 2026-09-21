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

#ifndef DALI_C_API_2_PIPELINE_REGISTRY_H_
#define DALI_C_API_2_PIPELINE_REGISTRY_H_

#include <cstddef>
#include <mutex>
#include <unordered_map>
#include "dali/core/api_helper.h"

namespace dali::c_api {

/** Tracks pipeline instances handed out through the C APIs.
 *
 * Every pipeline object created by either the legacy or the new C API is registered here
 * together with a function that destroys it. This makes it possible to tear down pipelines
 * that the user has leaked when the library is shut down, instead of letting them run past
 * the lifetime of the resources they depend on.
 */
class DLL_PUBLIC PipelineRegistry {
 public:
  using Deleter = void (*)(void *);

  static PipelineRegistry &instance();

  void Register(void *pipeline, Deleter deleter);

  /** Removes the pipeline from the registry without destroying it. */
  void Unregister(void *pipeline);

  /** Destroys all registered pipelines and empties the registry.
   *
   * @return the number of pipelines that were destroyed
   */
  size_t DestroyAll();

  size_t Count() const;

 private:
  PipelineRegistry() = default;

  mutable std::mutex mtx_;
  std::unordered_map<void *, Deleter> pipelines_;
};

/** Returns the number of pipeline instances created by the C APIs that were not destroyed. */
DLL_PUBLIC size_t GetOutstandingPipelineCount();

/** Destroys all pipeline instances created by the C APIs that were not destroyed by the user.
 *
 * @return the number of pipelines that were destroyed
 */
DLL_PUBLIC size_t DestroyOutstandingPipelines();

}  // namespace dali::c_api

#endif  // DALI_C_API_2_PIPELINE_REGISTRY_H_
