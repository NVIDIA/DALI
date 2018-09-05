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

#include "dali/pipeline/data/global_workspace.h"

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <utility>

#include "dali/pipeline/data/allocator_manager.h"

namespace dali {

std::map<int, std::mutex> mutexes_;

GlobalWorkspace::GlobalDeviceWorkspace *GlobalWorkspace::GetDeviceWorkspace() {
  int device;
  auto err = cudaGetDevice(&device);

  if (err == cudaErrorCudartUnloading) {
    // Application teardown,
    // just leave
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(workspace_mutex_);

  // Lazily allocate new workspaces as new devices come in
  if (!workspaces_.count(device)) {
    workspaces_[device] =
      std::unique_ptr<GlobalDeviceWorkspace>(new GlobalDeviceWorkspace(
            device,
            default_allocator_mgr_));
  }

  return workspaces_[device].get();
}

// Actual allocations
void *GlobalWorkspace::GlobalDeviceWorkspace::AllocateHost(const size_t bytes,
                                                           const bool pinned) const {
  void *ptr = nullptr;
  if (!pinned) {
    alloc_mgr_.GetCPUAllocator().New(&ptr, bytes);
  } else {
    alloc_mgr_.GetPinnedCPUAllocator().New(&ptr, bytes);
  }
  return ptr;
}

void GlobalWorkspace::GlobalDeviceWorkspace::FreeHost(void *ptr,
                                                      const size_t bytes,
                                                      const bool pinned) const {
  if (!pinned) {
    alloc_mgr_.GetCPUAllocator().Delete(ptr, bytes);
  } else {
    alloc_mgr_.GetPinnedCPUAllocator().Delete(ptr, bytes);
  }
}

template <>
unique_ptr<Buffer<CPUBackend>>
GlobalWorkspace::GlobalDeviceWorkspace::AcquireBuffer(size_t size, bool pinned) {
  return buffer_manager_->AcquireBuffer(size, pinned);
}

template <>
unique_ptr<Buffer<GPUBackend>>
GlobalWorkspace::GlobalDeviceWorkspace::AcquireBuffer(size_t size, bool /* unused */) {
  return buffer_manager_->AcquireBuffer(size);
}

template <>
void
GlobalWorkspace::GlobalDeviceWorkspace::ReleaseBuffer(unique_ptr<Buffer<CPUBackend>> *buffer,
                                                      bool pinned) {
  buffer_manager_->ReleaseBuffer(buffer, pinned);
}

template <>
void
GlobalWorkspace::GlobalDeviceWorkspace::ReleaseBuffer(unique_ptr<Buffer<GPUBackend>> *buffer,
                                                      bool /* unused */) {
  buffer_manager_->ReleaseBuffer(buffer);
}

GlobalWorkspace &GlobalWorkspace::Get() {
  static GlobalWorkspace gw;
  return gw;
}

void GlobalWorkspace::Init(const OpSpec &cpu_allocator,
                           const OpSpec &pinned_cpu_allocator,
                           const OpSpec &gpu_allocator) {
  std::lock_guard<std::mutex> lock(init_mutex_);
  DALI_ENFORCE(init_ == false,
      "GlobalWorkspace may be initialized only once!");
  default_allocator_mgr_.SetCPUAllocator(cpu_allocator);
  default_allocator_mgr_.SetPinnedCPUAllocator(pinned_cpu_allocator);
  default_allocator_mgr_.SetGPUAllocator(gpu_allocator);
  init_ = true;
}

}  // namespace dali
