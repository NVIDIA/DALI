#ifndef NDLL_PIPELINE_STREAM_POOL_H_
#define NDLL_PIPELINE_STREAM_POOL_H_

#include <cuda_runtime_api.h>

#include "ndll/common.h"
#include "ndll/error_handling.h"

namespace ndll {

/**
 * @brief Keeps track of a pool of streams for the pipeline. Provides methods 
 * for the pipelie to enforce correct syncrhonization behavior on the main stream
 */
class StreamPool {
public:
/**
 * @brief StreamPool has a 'main_stream'. The StreamPool provides
 * methods to enforce expected behavior if all work was issued in
 * this stream. If `non-blocking` == true, all streams are allocated
 * as non-blocking streams.
 */
  StreamPool(cudaStream_t main_stream, int max_streams, bool non_blocking) :
    streams_({main_stream}), max_streams_(max_streams),
    non_blocking_(non_blocking ? cudaStreamNonBlocking : cudaStreamDefault) {}

  ~StreamPool() {
    for (int i = 1; i < streams_.size(); ++i) {
      CUDA_ENFORCE(cudaStreamDestroy(streams_[i]));
    }
    for (auto &event : events_) {
      CUDA_ENFORCE(cudaEventDestroy(event));
    }
  }

  /**
   * @brief Returns the maximum amount of streams allowed
   */
  vector<cudaStream_t> GetMaxStreams() {
    int num_new_streams = max_streams_ - streams_.size();
    for (int i = 0; i < num_new_streams; ++i) {
      cudaStream_t new_stream;
      CUDA_ENFORCE(cudaStreamCreateWithFlags(&new_stream, non_blocking_));
      streams_.push_back(new_stream);
    }
    return streams_;
  }

  /**
   * @brief Returns the main stream
   */
  cudaStream_t GetStream() {
    return streams_[0];
  }

  /**
   * @brief Inserts events and StreamWaitEvents into the main
   * stream to ensure all future work in the main stream waits 
   * for all work issued in this pool to complete
   */
  void SetMainStreamEvents() {
    int num_new_event = (streams_.size() - 1) - events_.size();
    for (int i = 0; i < num_new_event; ++i) {
      cudaEvent_t new_event;
      CUDA_ENFORCE(cudaEventCreateWithFlags(&new_event,
              cudaEventDisableTiming));
      events_.push_back(new_event);
    }

    // Note: We will have to see if calling 'StreamWaitEvent()' on
    // every stream other than the main is costly
    for (int i = 1; i < streams_.size(); ++i) {
      CUDA_ENFORCE(cudaEventRecord(events_[i-1], streams_[i]));
      CUDA_ENFORCE(cudaStreamWaitEvent(streams_[0], events_[i-1], 0));
    }
  }

  DISABLE_COPY_ASSIGN(StreamPool);
private:
  vector<cudaStream_t> streams_;
  vector<cudaEvent_t> events_;

  int max_streams_;
  int non_blocking_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_STREAM_POOL_H_
