// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_DATA_GLOBAL_WORKSPACE_H_
#define DALI_PIPELINE_DATA_GLOBAL_WORKSPACE_H_

#include <map>
#include <memory>
#include <set>
#include <vector>

#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/data/allocator_manager.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/buffer_manager.h"

namespace dali {

/**
 * @brief Class containing all global parameters,
 * shared by all Pipelines.
 */
class GlobalWorkspace {
 public:
  static GlobalWorkspace &Get();

  ~GlobalWorkspace() = default;

  void Init(const OpSpec &cpu_allocator,
            const OpSpec &pinned_cpu_allocator,
            const OpSpec &gpu_allocator);

  void *AllocateHost(const size_t bytes, const bool pinned) {
    const auto * g = GetDeviceWorkspace();
    DALI_ENFORCE(g != nullptr,
        "Unknown error during memory allocation.");
    return g->AllocateHost(bytes, pinned);
  }

  void FreeHost(void *ptr, const size_t bytes, const bool pinned) {
    const auto * g = GetDeviceWorkspace();
    if (g == nullptr) {
      // Application teardown,
      // just leave
      return;
    }
    return g->FreeHost(ptr, bytes, pinned);
  }

  void *AllocateGPU(const size_t bytes, const bool pinned = false) {
    const auto * g = GetDeviceWorkspace();
    DALI_ENFORCE(g != nullptr,
        "Unknown error during memory allocation.");
    return g->AllocateGPU(bytes, pinned);
  }
  void FreeGPU(void *ptr, const size_t bytes, const bool pinned = false) {
    const auto * g = GetDeviceWorkspace();
    if (g == nullptr) {
      // Application teardown,
      // just leave
      return;
    }
    return g->FreeGPU(ptr, bytes, pinned);
  }

  template <typename Backend>
  unique_ptr<Buffer<Backend>> AcquireBuffer(size_t size, bool pinned) {
    auto * g = GetDeviceWorkspace();
    DALI_ENFORCE(g != nullptr,
        "Unknown error during memory allocation.");
    return g->AcquireBuffer<Backend>(size, pinned);
  }

  template <typename Backend>
  void ReleaseBuffer(unique_ptr<Buffer<Backend>> *buffer, bool pinned) {
    auto * g = GetDeviceWorkspace();
    if (g == nullptr) {
      // Application teardown,
      // just leave
      return;
    }
    return g->ReleaseBuffer<Backend>(buffer, pinned);
  }

 private:
  /**
   * @brief Class containing all data shared
   * by Pipelines using the same device, for
   * example GPU memory pool.
   */
  class GlobalDeviceWorkspace {
   public:
    GlobalDeviceWorkspace(int device, const AllocatorManager &mgr) :
      device_(device),
      alloc_mgr_(mgr) {
      buffer_manager_.reset(new LinearBufferManager(device));
    }

    ~GlobalDeviceWorkspace() = default;

    void *AllocateHost(const size_t bytes, const bool pinned) const;

    void FreeHost(void *ptr, const size_t bytes, const bool pinned) const;

    void *AllocateGPU(const size_t bytes, const bool pinned = false) const;
    void FreeGPU(void *ptr, const size_t bytes, const bool pinned = false) const;

    template <typename Backend>
    unique_ptr<Buffer<Backend>> AcquireBuffer(size_t size, bool pinned);

    template <typename Backend>
    void ReleaseBuffer(unique_ptr<Buffer<Backend>> *buffer, bool pinned);

   private:
    int device_;

    AllocatorManager alloc_mgr_;
    // Buffer manager
    unique_ptr<BufferManagerBase> buffer_manager_;
  };

  GlobalWorkspace() : init_(false) {}
  GlobalDeviceWorkspace *GetDeviceWorkspace();

  AllocatorManager default_allocator_mgr_;
  std::map<int, std::unique_ptr<GlobalDeviceWorkspace>> workspaces_;
  std::mutex workspace_mutex_;
  std::mutex init_mutex_;
  bool init_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_GLOBAL_WORKSPACE_H_
