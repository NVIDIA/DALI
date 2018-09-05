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

#include "dali/pipeline/data/backend.h"

#include <mutex>
#include <memory>
#include <unordered_map>
#include <utility>

#include "dali/common.h"
#include "dali/pipeline/data/allocator.h"
#include "dali/pipeline/data/allocator_manager.h"
#include "dali/pipeline/data/global_workspace.h"
#include "dali/pipeline/operators/op_spec.h"

namespace dali {

// Sets the allocator ptrs for all backends
void InitializeBackends(const OpSpec &cpu_allocator,
    const OpSpec &pinned_cpu_allocator,
    const OpSpec &gpu_allocator) {
  GlobalWorkspace::Get().Init(cpu_allocator, pinned_cpu_allocator, gpu_allocator);
}

void* GPUBackend::New(size_t bytes, bool) {
  return GlobalWorkspace::Get().AllocateGPU(bytes);
}

void GPUBackend::Delete(void *ptr, size_t bytes, bool) {
  GlobalWorkspace::Get().FreeGPU(ptr, bytes);
}

void* CPUBackend::New(size_t bytes, bool pinned) {
  return GlobalWorkspace::Get().AllocateHost(bytes, pinned);
}

void CPUBackend::Delete(void *ptr, size_t bytes, bool pinned) {
  return GlobalWorkspace::Get().FreeHost(ptr, bytes, pinned);
}

}  // namespace dali
