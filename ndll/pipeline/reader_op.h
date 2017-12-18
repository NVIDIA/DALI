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
  prefetch_success_(true),
  finished_(false) {
    // TODO(slayton): Anything needed here?
  }

  virtual ~DataReader() noexcept {
    // check we're good
  }

  // perform the prefetching operation
  virtual bool Prefetch() = 0;

  // Main prefetch work loop
  void PrefetchWorker() {
    while (!finished_) {
      try {
        prefetch_success_ = Prefetch();
      } catch (const std::exception& e) {
        // error out

        // notify of failure
        prefetch_success_ = false;
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
      finished_ = true;
      // join the prefetch thread and destroy it
      prefetch_thread_->join();
      prefetch_thread_.reset();

    } else {
      finished_ = true;
    }
  }

  void Run(SampleWorkspace* ws) override {
    StartPrefetchThread();

    // consume batch
    Operator<Backend>::Run(ws);
  }

  void Run(DeviceWorkspace* ws) override {
    StartPrefetchThread();

    Operator<Backend>::Run(ws);
  }

 protected:
  std::unique_ptr<std::thread> prefetch_thread_;

  // check if prefetching was successful
  std::atomic<bool> prefetch_success_;

  // signal that the prefetch thread has finished
  std::atomic<bool> finished_;
};

};  // namespace ndll

#endif  // NDLL_PIPELINE_DATA_READER_OP_H_
