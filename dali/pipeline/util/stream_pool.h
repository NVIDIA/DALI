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

#ifndef DALI_PIPELINE_UTIL_STREAM_POOL_H_
#define DALI_PIPELINE_UTIL_STREAM_POOL_H_

#include <cuda_runtime_api.h>
#include <map>
#include <vector>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/util/device_guard.h"

namespace dali {

/**
 * @brief Manages the lifetimes and allocations of cuda streams.
 */
class StreamPool {
 public:
  /**
   * @brief Creates a pool with the given max size. If the input
   * size is < 0, the pool has no size limit.
   */
  explicit inline StreamPool(int max_size, bool non_blocking = true,
                             int default_cuda_stream_priority = 0)
      : max_size_(max_size),
        non_blocking_(non_blocking),
        default_cuda_stream_priority_(default_cuda_stream_priority) {
    DALI_ENFORCE(max_size != 0, "Stream pool must have non-zero size.");
  }

  inline ~StreamPool() noexcept(false) {
    for (auto &stream : streams_) {
      int device = stream_devices_[stream];
      DeviceGuard g(device);

      CUDA_CALL(cudaStreamSynchronize(stream));
      CUDA_CALL(cudaStreamDestroy(stream));
    }
  }

  /**
   * @brief Returns a stream from the pool. If max_size has been exceeded,
   * we hand out previously allocated streams round-robin.
   */
  cudaStream_t GetStream() {
    if (max_size_ < 0 || (Index)streams_.size() < max_size_) {
      cudaStream_t new_stream;
      // Note: Why is device tracked? Is StreamPool intended to be used across devices?
      int dev;
      cudaGetDevice(&dev);
      int flags = non_blocking_ ? cudaStreamNonBlocking : cudaStreamDefault;
      CUDA_CALL(cudaStreamCreateWithPriority(&new_stream, flags, default_cuda_stream_priority_));
      streams_.push_back(new_stream);
      stream_devices_[new_stream] = dev;
      return new_stream;
    }
    cudaStream_t stream = streams_[idx_];
    idx_ = (idx_+1) % streams_.size();
    return stream;
  }

 private:
  vector<cudaStream_t> streams_;
  // track which streams are on which devices
  std::map<cudaStream_t, int> stream_devices_;

  int max_size_, idx_ = 0;
  bool non_blocking_;
  int default_cuda_stream_priority_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_STREAM_POOL_H_
