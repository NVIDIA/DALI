// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/backend.h"
#include "dali/core/mm/memory.h"

namespace dali {

// this is to make debug builds happy about kMaxGrowthFactor
template class Buffer<CPUBackend>;
template class Buffer<GPUBackend>;


DLL_PUBLIC shared_ptr<uint8_t> AllocBuffer(size_t bytes, bool /* pinned */, GPUBackend *) {
  const size_t kDevAlignment = 256;  // warp alignment for 32x64-bit
  return mm::alloc_raw_shared<uint8_t, mm::memory_kind::device>(bytes, kDevAlignment);
}

DLL_PUBLIC shared_ptr<uint8_t> AllocBuffer(size_t bytes, bool pinned, CPUBackend *) {
  const size_t kHostAlignment = 64;  // cache alignment
  if (pinned)
    return mm::alloc_raw_shared<uint8_t, mm::memory_kind::pinned>(bytes, kHostAlignment);
  else
    return mm::alloc_raw_shared<uint8_t, mm::memory_kind::host>(bytes, kHostAlignment);
}

}  // namespace dali
