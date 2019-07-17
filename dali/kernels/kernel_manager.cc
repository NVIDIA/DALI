// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/kernels/kernel_manager.h"

namespace dali {
namespace kernels {

void KernelManager::Initialize(size_t num_instances, size_t num_threads) {
  instances.resize(num_instances);
  scratchpads.resize(num_threads);
}

void KernelManager::Reset() {
  instances.clear();
  scratchpads.clear();
  max_scratch_sizes = {};
}

void KernelManager::ReserveScratchpad(
    ScratchpadAllocator &sa,
    const ScratchSizes &sizes) {
  size_t N = ScratchpadAllocator::NumAllocTypes;

  // is scratchpad big enough?
  auto is_big_enough = [&]() {
    auto caps = sa.Capacities();
    for (size_t i = 0; i < caps.size(); i++)
      if (caps[i] < sizes[i])
        return false;
    return true;
  };
  // if the scratchpad happens to be big enough, then just return
  if (is_big_enough())
    return;

  for (size_t i = 0; i < sizes.size(); i++) {
    if (sizes[i] > max_scratch_sizes[i])
      max_scratch_sizes[i] = sizes[i];
  }

  sa.Reserve(max_scratch_sizes);
}


}  // namespace kernels
}  // namespace dali
