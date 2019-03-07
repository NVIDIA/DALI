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

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/util/device_guard.h"

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
  explicit inline EventPool(int max_size) : max_size_(max_size) {
    DALI_ENFORCE(max_size != 0, "Event pool must have non-zero size.");
  }

  inline ~EventPool() noexcept(false) {
    for (auto &event : events_) {
      DeviceGuard g(event_devices_[event]);
      CUDA_CALL(cudaEventSynchronize(event));
      CUDA_CALL(cudaEventDestroy(event));
    }
  }

  /**
   * @brief Returns a event from the pool. If max_size has been exceeded,
   * we hand out previously allocated events round-robin.
   */
  cudaEvent_t GetEvent() {
    if (max_size_ < 0 || (Index)events_.size() < max_size_) {
      cudaEvent_t new_event;
      CUDA_CALL(cudaEventCreateWithFlags(&new_event, cudaEventDisableTiming));
      events_.push_back(new_event);

      int dev;
      CUDA_CALL(cudaGetDevice(&dev));
      event_devices_[new_event] = dev;

      return new_event;
    }
    cudaEvent_t event = events_[idx_];
    idx_ = (idx_+1) % events_.size();
    return event;
  }

 private:
  vector<cudaEvent_t> events_;
  std::map<cudaEvent_t, int> event_devices_;
  int max_size_, idx_ = 0;
};

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_EVENT_POOL_H_
