// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_UTIL_EVENT_POOL_H_
#define NDLL_PIPELINE_UTIL_EVENT_POOL_H_

#include <cuda_runtime_api.h>
#include <vector>

#include "ndll/common.h"
#include "ndll/error_handling.h"

namespace ndll {

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
    NDLL_ENFORCE(max_size != 0, "Event pool must have non-zero size.");
  }

  inline ~EventPool() {
    for (auto &event : events_) {
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
      return new_event;
    }
    cudaEvent_t event = events_[idx_];
    idx_ = (idx_+1) % events_.size();
    return event;
  }

 private:
  vector<cudaEvent_t> events_;
  int max_size_, idx_ = 0;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_UTIL_EVENT_POOL_H_
