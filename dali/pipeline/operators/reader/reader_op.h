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

#ifndef DALI_PIPELINE_OPERATORS_READER_READER_OP_H_
#define DALI_PIPELINE_OPERATORS_READER_READER_OP_H_

#include <atomic>
#include <condition_variable>
#include <memory>
#include <thread>
#include <vector>

#include "dali/pipeline/operators/reader/loader/loader.h"
#include "dali/pipeline/operators/reader/parser/parser.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

/**
 * @brief BaseClass for operators that perform prefetching work
 *
 * Operator runs an additional prefetch thread
 */

/**
 * @brief  BaseClass for operators that perform prefetching work
 *
 * Operator runs an additional prefetch thread
 * @tparam Backend
 * @tparam LoadTarget Type that Loader will load data into, used also to store prefetched
 *                    samples.
 * @tparam ParseTarget Type passed into Parser for parsing, usually it is the same
 *                     as the LoadTarget.
 */
template <typename Backend, typename LoadTarget, typename ParseTarget = LoadTarget>
class DataReader : public Operator<Backend> {
 public:
  inline explicit DataReader(const OpSpec& spec) :
    Operator<Backend>(spec),
  thread_locks_(Operator<Backend>::num_threads_),
  condition_vars_(Operator<Backend>::num_threads_),
  prefetch_ready_(false),
  prefetch_ready_workers_(false),
  prefetch_success_(true),
  finished_(false),
  samples_processed_(0),
  batch_stop_(false) {
    // TODO(slayton): Anything needed here?
  }

  virtual ~DataReader() noexcept {
    StopPrefetchThread();
  }

  // perform the prefetching operation
  virtual bool Prefetch() {
    // first clear the batch
    // TODO(slayton): exchange multiple batches
    prefetched_batch_.clear();

    for (int i = 0; i < Operator<Backend>::batch_size_; ++i) {
      auto* t = loader_->ReadOne();
      prefetched_batch_.push_back(t);
    }

    // TODO(slayton): swap prefetched batches around

    return true;
  }

  // Main prefetch work loop
  void PrefetchWorker() {
    std::unique_lock<std::mutex> lock(prefetch_access_mutex_);

    // if a result is already ready, wait until it's consumed
    while (prefetch_ready_) {
      producer_.wait(lock);
    }

    while (!finished_) {
      try {
        prefetched_batch_.reserve(Operator<Backend>::batch_size_);
        prefetch_success_ = Prefetch();
      } catch (const std::exception& e) {
        printf("Prefetch Failed\n");
        DALI_FAIL(e.what());
        // notify of failure
        prefetch_success_ = false;
      }
      // mark as ready
      prefetch_ready_ = true;
      // notify the consumer of a result ready to consume
      consumer_.notify_all();

      // wait until the result is consumed
      while (prefetch_ready_) {
        producer_.wait(lock);
      }
    }
  }

  // to be called in constructor
  void StartPrefetchThread() {
    {
      // if thread hasn't been started yet, start it
      if (!prefetch_thread_.get()) {
        prefetch_thread_.reset(
            new std::thread([this] { this->PrefetchWorker(); }));
      }
    }
  }

  // to be called in destructor
  void StopPrefetchThread() {
    if (prefetch_thread_.get()) {
      {
        std::unique_lock<std::mutex> lock(prefetch_access_mutex_);

        while (!prefetch_ready_) {
          consumer_.wait(lock);
        }
        finished_ = true;
        prefetch_ready_ = false;
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

  void Run(SampleWorkspace* ws) override {
    {
      std::unique_lock<std::mutex> lock(prefetch_access_mutex_);
      StartPrefetchThread();

      if (batch_stop_) batch_stop_ = false;
    }

    {
      // block all other worker threads from taking the prefetch-controller lock
      std::unique_lock<std::mutex> worker_lock(worker_mutex_);

      if (!prefetch_ready_workers_) {
        // grab the actual prefetching lock
        std::unique_lock<std::mutex> prefetch_lock(prefetch_access_mutex_);

        // Wait until prefetch is ready
        while (!prefetch_ready_) {
          consumer_.wait(prefetch_lock);
          prefetch_ready_ = true;
        }
        // signal the other workers we're ready
        prefetch_ready_workers_ = true;

        // signal the prefetch thread to start again
        prefetch_ready_ = true;
        producer_.notify_one();
      }
    }

    // consume batch
    Operator<Backend>::Run(ws);

    loader_->ReturnTensor(prefetched_batch_[ws->data_idx()]);

    samples_processed_++;

    // lock, check if batch is finished, notify
    {
      std::unique_lock<std::mutex> lock(prefetch_access_mutex_);

      // if we need to stop, stop.
      if (batch_stop_) return;

      // if we've consumed all samples in this batch, reset state and stop
      if (samples_processed_.load() == Operator<Backend>::batch_size_) {
        prefetch_ready_workers_ = false;
        prefetch_ready_ = false;
        producer_.notify_one();
        samples_processed_ = 0;
        batch_stop_ = true;
      }
      return;
    }
  }

  Index epoch_size() const override {
    return loader_->Size();
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
  std::vector<LoadTarget*> prefetched_batch_;

  // keep track of how many samples have been processed
  // over all threads.
  std::atomic<int> samples_processed_;

  // notify threads to stop processing
  std::atomic<bool> batch_stop_;

  // Loader
  std::unique_ptr<Loader<Backend, LoadTarget>> loader_;

  // Parser
  std::unique_ptr<Parser<ParseTarget>> parser_;
};

#define USE_READER_OPERATOR_MEMBERS_1(Backend, LoadTarget) \
  using DataReader<Backend, LoadTarget>::loader_;          \
  using DataReader<Backend, LoadTarget>::parser_;          \
  using DataReader<Backend, LoadTarget>::prefetched_batch_;

#define USE_READER_OPERATOR_MEMBERS_2(Backend, LoadTarget, ParseTarget) \
  using DataReader<Backend, LoadTarget, ParseTarget>::loader_;          \
  using DataReader<Backend, LoadTarget, ParseTarget>::parser_;          \
  using DataReader<Backend, LoadTarget, ParseTarget>::prefetched_batch_;

#define USE_READER_OPERATOR_MEMBERS(Backend, ...) \
  GET_MACRO(__VA_ARGS__,                          \
            USE_READER_OPERATOR_MEMBERS_2,        \
            USE_READER_OPERATOR_MEMBERS_1)(Backend, __VA_ARGS__)

};  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_READER_OP_H_
