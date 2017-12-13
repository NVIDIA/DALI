#ifndef NDLL_PIPELINE_OPERATORS_PREFETCHED_DATA_READER_OPERATOR_H_
#define NDLL_PIPELINE_OPERATORS_PREFETCHED_DATA_READER_OPERATOR_H_

#include <atomic>
#include <condition_variable>
#include <memory>
#include <thread>

#include "ndll/pipeline/data_store/data_store.h"
#include "ndll/pipeline/data_store/lmdb.h"
#include "ndll/pipeline/operator.h"

namespace ndll {

/**
 * @brief BaseClass for operators that perform prefetching work
 *
 * Operator runs an additional prefetch thread
 */
template <typename Backend>
class DataReaderOperator : public Operator<Backend> {
 public:
  inline explicit DataReaderOperator(const OpSpec& spec) :
    Operator<Backend>(spec) {
    // TODO() stuff here
    //
  }

  virtual ~DataReaderOperator() noexcept {
    // check we're good
    StopPrefetchThread();
  }

  // perform the prefetching operation
  virtual bool Prefetch() = 0;

  // Main prefetch work loop
  void PrefetchWorker() {
    std::unique_lock<std::mutex> lock(prefetch_access_mutex_);

    // if a result is already ready, wait until it's consumed
    while (prefetch_ready_) {
      producer_.wait(lock);
    }

    while (!finished_) {
      try {
        prefetch_success_ = Prefetch();
      } catch (const std::exception& e) {
        // error out

        // notify of failure
        prefetch_success_ = false;
      }
      // mark as ready
      prefetch_ready_ = true;
      // notify the consumer of a result ready to consume
      consumer_.notify_one();

      // wait until the result is consumed
      while (prefetch_ready_) {
        producer_.wait(lock);
      }
    }
  }

  // to be called in constructor
  void StartPrefetchThread() {
    // if thread hasn't been started yet, start it
    if (!prefetch_thread_.get()) {
      prefetch_thread_.reset(
          new std::thread([this] { this->PrefetchWorker(); }));
    }
  }

  // to be called in destructor
  void StopPrefetchThread() {
    if (prefetch_thread_.get()) {
      std::unique_lock<std::mutex> lock(prefetch_access_mutex_);

      while (!prefetch_ready_) {
        consumer_.wait(lock);
      }
      // notify the prefetcher to stop
      producer_.notify_one();
      // join the prefetch thread and destroy it
      prefetch_thread_->join();
      prefetch_thread_.reset();

    } else {
      finished_ = true;
    }
  }

 protected:
  std::unique_ptr<std::thread> prefetch_thread_;

  // mutex to control access to the producer
  std::mutex prefetch_access_mutex_;

  // signals for producer and consumer
  std::condition_variable producer_, consumer_;

  // signal that a complete batch has been prefetched
  std::atomic<bool> prefetch_ready_;

  // check if prefetching was successful
  std::atomic<bool> prefetch_success_;

  // signal that the prefetch thread has finished
  std::atomic<bool> finished_;
};

};  // namespace ndll

#endif
