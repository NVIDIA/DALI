// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/cuda_error.h"
#include "dali/core/cuda_stream.h"
#include "dali/core/cuda_event_pool.h"
#include "dali/core/device_guard.h"

namespace dali {

AccessOrder::AccessOrder(cudaStream_t stream) : stream_(stream) {
  if (is_device())
    device_id_ = DeviceFromStream(stream);
}

void AccessOrder::wait(const AccessOrder &other) const {
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

    auto current_dev = []() {
      int dev;
      CUDA_CALL(cudaGetDevice(&dev));
      return dev;
    };

    // If the stream handle has a special value, we can't refer to it directly - it is
    // inherently associated with the concept of "current device" and it must be switched
    if (other_dev != device_id_ ||
        ((other.stream_ == 0 ||
         other.stream_ == cudaStreamPerThread ||
         other.stream_ == cudaStreamLegacy) &&
         other_dev != current_dev())) {
      DeviceGuard dg(other.device_id_);
      CUDA_CALL(cudaEventRecord(event, other.stream()));
    } else {
      CUDA_CALL(cudaEventRecord(event, other.stream()));
    }
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
