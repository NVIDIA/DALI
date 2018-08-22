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

#include "dali/pipeline/data/allocator_manager.h"

#include <map>
#include <memory>
#include <mutex>

#include "dali/pipeline/data/global_workspace.h"

namespace dali {

unique_ptr<CPUAllocator> AllocatorManager::cpu_allocator_(nullptr);
unique_ptr<CPUAllocator> AllocatorManager::pinned_cpu_allocator_(nullptr);
unique_ptr<GPUAllocator> AllocatorManager::gpu_allocator_(nullptr);
std::mutex AllocatorManager::mutex_;

}  // namespace dali
