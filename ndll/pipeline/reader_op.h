// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_DATA_READER_OP_H_
#define NDLL_PIPELINE_DATA_READER_OP_H_

#include <atomic>
#include <condition_variable>
#include <memory>
#include <thread>

#include "ndll/pipeline/loader/loader.h"
#include "ndll/pipeline/operator.h"

namespace ndll {

/**
 * @brief BaseClass for operators that perform prefetching work
 *
 * Operator runs an additional prefetch thread
 */
template <typename Backend>
class DataReader : public Operator<Backend> {
 public:
  inline explicit DataReader(const OpSpec& spec) :
    Operator<Backend>(spec),
  prefetch_ready_(false),
  prefetch_success_(true),
  finished_(false) {
    // TODO(slayton): Anything needed here?
  }

  virtual ~DataReader() noexcept {
    printf("calling ~DataReader\n");
    // check we're good
    // StopPrefetchThread();
  }

  // perform the prefetching operation
  virtual bool Prefetch() = 0;

  // Main prefetch work loop
  void PrefetchWorker() {
    std::unique_lock<std::mutex> lock(prefetch_access_mutex_);

    printf("started prefetch worker\n");
    // if a result is already ready, wait until it's consumed
    while (prefetch_ready_) {
      producer_.wait(lock);
    }

    while (!finished_) {
      try {
        printf("calling Prefetch()\n");
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
      printf("waiting for consumption\n");
      while (prefetch_ready_) {
        producer_.wait(lock);
      }
      printf("consumed\n");
    }
  }

  // to be called in constructor
  void StartPrefetchThread() {
    // if thread hasn't been started yet, start it
    if (!prefetch_thread_.get()) {
      prefetch_thread_.reset(
          new std::thread([this] { this->PrefetchWorker(); }));
    }
    printf("Prefetch thread started\n");
  }

  // to be called in destructor
  void StopPrefetchThread() {
    if (prefetch_thread_.get()) {
      {
        std::unique_lock<std::mutex> lock(prefetch_access_mutex_);

        printf("waiting on consumer\n");
        while (!prefetch_ready_) {
          consumer_.wait(lock);
        }
        finished_ = true;
        prefetch_ready_ = false;
        printf("done\n");
      }
      // notify the prefetcher to stop
      producer_.notify_one();
      // join the prefetch thread and destroy it
      printf("joining prefetch thread\n");
      prefetch_thread_->join();
      printf("joined\n");
      prefetch_thread_.reset();

    } else {
      finished_ = true;
    }
  }

  void Run(SampleWorkspace* ws) override {
    std::unique_lock<std::mutex> lock(prefetch_access_mutex_);
    StartPrefetchThread();

    // wait for a batch to be ready
    while (!prefetch_ready_) {
      consumer_.wait(lock);
    }

    printf("running RunPerSampleCPU\n");
    // consume batch
    Operator<Backend>::Run(ws);
    printf("finished RunPerSampleCPU\n");

    prefetch_ready_ = false;
    producer_.notify_one();
  }

  void Run(DeviceWorkspace* ws) override {
    std::unique_lock<std::mutex> lock(prefetch_access_mutex_);
    StartPrefetchThread();

    // wait for a batch to be ready
    while (!prefetch_ready_) {
      consumer_.wait(lock);
    }

    Operator<Backend>::Run(ws);

    prefetch_ready_ = false;
    producer_.notify_one();
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

#endif  // NDLL_PIPELINE_DATA_READER_OP_H_
