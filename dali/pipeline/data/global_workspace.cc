// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/pipeline/data/global_workspace.h"

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <utility>

#include "dali/pipeline/data/allocator_manager.h"

namespace dali {

std::map<int, std::mutex> mutexes_;
std::map<int, std::unique_ptr<GlobalWorkspace>> workspaces_;

GlobalWorkspace *GlobalWorkspace::Get(int device) {
  // Will automagically call default constructor under the hood for
  // un-initialised entries.
  std::lock_guard<std::mutex> lock(mutexes_[device]);

  // Lazily allocate new workspaces as new devices come in
  if (!workspaces_.count(device)) {
    workspaces_[device] = std::unique_ptr<GlobalWorkspace>(new GlobalWorkspace(device));
  }

  return workspaces_[device].get();
}

GlobalWorkspace *GlobalWorkspace::Get() {
  int device;
  auto err = cudaGetDevice(&device);

  if (err == cudaErrorCudartUnloading) {
    return nullptr;
  }

  return GlobalWorkspace::Get(device);
}

GlobalWorkspace::GlobalWorkspace(int device)
  : device_(device) {
    buffer_manager_.reset(new LinearBufferManager(device));
}

}  // namespace dali
