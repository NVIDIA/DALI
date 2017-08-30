#ifndef NDLL_STREAM_POOL_H_
#define NDLL_STREAM_POOL_H_

#include <cuda_runtime_api.h>

#include "ndll/common.h"

namespace ndll {

/**
 * @brief Keeps track of all the streams for the pipeline. 
 * Provides access for the ops, and provides methods for 
 * the pipeline to enforce correct synchronization behavior
 * on the main stream
 */
class StreamPool {
public:
  /**
   * @brief Stream pool has a 'main_stream'. The StreamPool 
   * provides method to enforce expected behavior if all 
   * work was issued in this stream. If `non_blocking` == true, 
   * all streams are allocated as non-blocking streams
   */
  StreamPool(cudaStream_t main_stream, int max_streams, bool non_blocking) :
    streams_({main_stream}), max_streams_(max_streams),
    stream_flags_(non_blocking ? cudaStreamNonBlocking : cudaStreamDefault) {}

  ~StreamPool() {
    // Cleanup all but the main stream
    for (int i = 1; i < streams_.size(); ++i) {
      CUDA_CALL(cudaStreamDestroy(streams_[i]));
    }
    for (auto &event : events_) {
      CUDA_CALL(cudaEventDestroy(event));
    }
  }

  /**
   * Get the maximum amount of streams allowed
   */
  vector<cudaStream_t> GetMaxStreams() {
    int num_new_streams = max_streams_ - streams_.size();
    for (int i = 0; i < num_new_streams; ++i) {
      cudaStream_t new_stream;
      CUDA_CALL(cudaStreamCreateWithFlags(&new_stream, stream_flags_));
      streams_.push_back(new_stream);
    }
    return streams_;
  }
  
  /** 
   * @brief Get the main stream
   */
  cudaStream_t Stream() {
    return streams_[0];
  }

  /**
   * @brief Inserts events and StreamWaitEvents to ensure all 
   * future work in the main stream waits for all work in this 
   * pool to complete
   */
  void SetMainStreamEvents() {
    int num_new_event = (streams_.size() - 1) - events_.size();
    for (int i = 0; i < num_new_event; ++i) {
      cudaEvent_t new_event;
      CUDA_CALL(cudaEventCreateWithFlags(&new_event,
              cudaEventDisableTiming));
      events_.push_back(new_event);
    }

    // Note: We will have to see if calling 'StreamWaitEvent()' on
    // every stream other than the main is costly
    for (int i = 1; i < streams_.size(); ++i) {
      CUDA_CALL(cudaEventRecord(events_[i-1], streams_[i]));
      CUDA_CALL(cudaStreamWaitEvent(streams_[0], events_[i-1], 0));
    }
  }

  DISABLE_COPY_ASSIGN(StreamPool);
private:
  vector<cudaStream_t> streams_;
  int max_streams_;
  unsigned int stream_flags_;

  vector<cudaEvent_t> events_;
};

} // namespace ndll

#endif // NDLL_STREAM_POOL_H_
