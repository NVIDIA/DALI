// Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

DLL_PUBLIC AccessOrder get_deletion_order(const std::shared_ptr<void> &ptr) {
  if (auto *del = std::get_deleter<mm::AsyncDeleter>(ptr))
    return AccessOrder(del->release_on_stream);
  else
    return {};
}

DLL_PUBLIC bool set_deletion_order(const std::shared_ptr<void> &ptr, AccessOrder order) {
  if (ptr.use_count() == 1 && order.has_value()) {
    if (auto *del = std::get_deleter<mm::AsyncDeleter>(ptr)) {
      del->release_on_stream = order.get();
      if (ptr.use_count() != 1)
        throw std::logic_error("Race condition detected - the pointer is no longer unique.");
      return true;
    }
  }
  return false;
}


DLL_PUBLIC shared_ptr<uint8_t> AllocBuffer(size_t bytes, bool /* device_ordinal */,
                                           int device_id,
                                           AccessOrder order,
                                           GPUBackend *) {
  const size_t kDevAlignment = 256;  // warp alignment for 32x64-bit
  cudaStream_t s = order.has_value() ? order.get() : AccessOrder::host_sync_stream();
  auto *rsrc = mm::GetDefaultDeviceResource(device_id);
  return mm::alloc_raw_async_shared<uint8_t>(rsrc, bytes, s, s, kDevAlignment);
}

DLL_PUBLIC shared_ptr<uint8_t> AllocBuffer(size_t bytes, bool pinned,
                                           int /* device_ordinal */,
                                           AccessOrder order,
                                           CPUBackend *) {
  const size_t kHostAlignment = 64;  // cache alignment
  if (pinned) {
    cudaStream_t s = order.has_value() ? order.get() : AccessOrder::host_sync_stream();
    return mm::alloc_raw_async_shared<uint8_t, mm::memory_kind::pinned>(
      bytes, s, s, kHostAlignment);
  } else {
    return mm::alloc_raw_shared<uint8_t, mm::memory_kind::host>(bytes, kHostAlignment);
  }
}

DLL_PUBLIC bool RestrictPinnedMemUsage() {
  static bool val = []() {
    const char *env = getenv("DALI_RESTRICT_PINNED_MEM");
    return env && atoi(env);
  }();
  return val;
}

// this is to make debug builds happy about kMaxGrowthFactor
template class Buffer<CPUBackend>;
template class Buffer<GPUBackend>;

}  // namespace dali
