// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_UTIL_STREAM_POOL_H_
#define NDLL_PIPELINE_UTIL_STREAM_POOL_H_

#include <cuda_runtime_api.h>
#include <vector>

#include "ndll/common.h"
#include "ndll/error_handling.h"

namespace ndll {

/**
 * @brief Manages the lifetimes and allocations of cuda streams.
 */
class StreamPool {
 public:
  /**
   * @brief Creates a pool with the given max size. If the input
   * size is < 0, the pool has no size limit.
   */
  explicit inline StreamPool(int max_size, bool non_blocking = true) :
    max_size_(max_size), non_blocking_(non_blocking) {
    NDLL_ENFORCE(max_size != 0, "Stream pool must have non-zero size.");
  }

  inline ~StreamPool() {
    for (auto &stream : streams_) {
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
      int dev;
      cudaGetDevice(&dev);
      CUDA_CALL(cudaStreamCreateWithFlags(&new_stream,
              non_blocking_ ? cudaStreamNonBlocking : cudaStreamDefault));
      streams_.push_back(new_stream);
      return new_stream;
    }
    cudaStream_t stream = streams_[idx_];
    idx_ = (idx_+1) % streams_.size();
    return stream;
  }

 private:
  vector<cudaStream_t> streams_;
  int max_size_, idx_ = 0;
  bool non_blocking_;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_UTIL_STREAM_POOL_H_
