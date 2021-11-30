// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/access_order.h"
#include "dali/core/cuda_event_pool.h"

namespace dali {

void AccessOrder::join(const AccessOrder &other) const {
  if (*this == other)
    return;
  // If this order is null, do nothing.
  // If the order to synchronize after is null or host, do nothing - host order is
  // always considered up-to-date.
  if (!has_value() || !other.is_device())
    return;
  if (is_device()) {
    auto &pool = CUDAEventPool::instance();
    int other_dev = other.device_id();
    auto event = pool.Get(other_dev);
    // Record an event in the preceding stream
    CUDA_CALL(cudaEventRecord(event, other.stream()));
    // and wait for it in this stream
    CUDA_CALL(cudaStreamWaitEvent(stream(), event, 0));
    pool.Put(std::move(event), other_dev);
  } else {
    // host order - wait for the preceding stream on host
    CUDA_CALL(cudaStreamSynchronize(other.stream()));
  }
}

void AccessOrder::wait(cudaEvent_t event) const {
  if (!has_value())
    throw std::logic_error("A null AccessOrder cannot wait for an event.");
  if (is_device()) {
    CUDA_DTOR_CALL(cudaStreamWaitEvent(stream(), event, 0));
  } else {
    CUDA_DTOR_CALL(cudaEventSynchronize(event));
  }
}

}  // namespace dali
