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

#ifndef DALI_PIPELINE_UTIL_EVENT_POOL_H_
#define DALI_PIPELINE_UTIL_EVENT_POOL_H_

#include <cuda_runtime_api.h>
#include <map>
#include <vector>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/device_guard.h"

namespace dali {

/**
 * @brief Manages the lifetimes and allocations of cuda events.
 */
class EventPool {
 public:
  /**
   * @brief Creates a pool with the given max size. If the input
   * size is < 0, the pool has no size limit.
   */
  inline EventPool() = default;

  inline ~EventPool() noexcept(false) {
    for (auto &event_info : events_) {
      DeviceGuard g(event_info.device);
      CUDA_CALL(cudaEventSynchronize(event_info.event));
      CUDA_CALL(cudaEventDestroy(event_info.event));
    }
  }

  /**
   * @brief Returns a event from the pool.
   */
  cudaEvent_t GetEvent() {
    int dev;
    CUDA_CALL(cudaGetDevice(&dev));

    cudaEvent_t new_event;
    CUDA_CALL(cudaEventCreateWithFlags(&new_event, cudaEventDisableTiming));
    events_.push_back({new_event, dev});

    return new_event;
  }

 private:
  /**
   * @brief Stores information about created event - the event and the device it is created on.
   */
  struct event_device_info {
    cudaEvent_t event;
    int device;
  };

  std::vector<event_device_info> events_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_EVENT_POOL_H_
