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
  prefetch_ready_workers_(false),
  prefetch_success_(true),
  finished_(false),
  samples_processed_(0),
  condition_vars_(Operator<Backend>::num_threads_),
  thread_locks_(Operator<Backend>::num_threads_) {
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
        prefetched_batch_.reserve(Operator<Backend>::batch_size_);
        prefetch_success_ = Prefetch();
      } catch (const std::exception& e) {
        // error out

        // notify of failure
        prefetch_success_ = false;
      }
      // mark as ready
      prefetch_ready_ = true;
      // notify the consumer of a result ready to consume
      consumer_.notify_all();

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
    {
      // if thread hasn't been started yet, start it
      if (!prefetch_thread_.get()) {
        prefetch_thread_.reset(
            new std::thread([this] { this->PrefetchWorker(); }));
        printf("Prefetch thread started\n");
      }
    }
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
    {
      std::unique_lock<std::mutex> lock(prefetch_access_mutex_);
      StartPrefetchThread();
    }

    {
      // block all other worker threads from taking the prefetch-controller lock
      std::unique_lock<std::mutex> worker_lock(worker_mutex_);
      if (!prefetch_ready_workers_) {

        std::unique_lock<std::mutex> prefetch_lock(prefetch_access_mutex_);
        while (!prefetch_ready_) {
          consumer_.wait(prefetch_lock);
          prefetch_ready_ = true;
        }
        // signal the other workers we're ready
        prefetch_ready_workers_ = true;

        // signal the prefetch thread to start again
        prefetch_ready_ = false;
        producer_.notify_one();
      }
    }

    printf("[%d] running RunPerSampleCPU\n", ws->thread_idx());
    // consume batch
    Operator<Backend>::Run(ws);
    printf("[%d] finished RunPerSampleCPU\n", ws->thread_idx());

    samples_processed_++;

    if (samples_processed_.load() == Operator<Backend>::batch_size_-1) {
      // lock, reset, notify
      std::unique_lock<std::mutex> lock(prefetch_access_mutex_);
      if (prefetch_ready_workers_) {
        prefetch_ready_workers_ = false;
        // producer_.notify_one();
      }
      return;
    }
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
  std::mutex worker_mutex_;
  std::vector<std::mutex> thread_locks_;

  // signals for producer and consumer
  std::condition_variable producer_, consumer_;
  std::vector<std::condition_variable> condition_vars_;
  std::condition_variable worker_threads_;

  // signal that a complete batch has been prefetched
  std::atomic<bool> prefetch_ready_;
  std::atomic<bool> prefetch_ready_workers_;

  // check if prefetching was successful
  std::atomic<bool> prefetch_success_;

  // signal that the prefetch thread has finished
  std::atomic<bool> finished_;

  // prefetched batch
  std::vector<Tensor<Backend>*> prefetched_batch_;

  // keep track of how many samples have been processed
  // over all threads.
  std::atomic<int> samples_processed_;
};

};  // namespace ndll

#endif  // NDLL_PIPELINE_DATA_READER_OP_H_
