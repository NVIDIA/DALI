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
#include <string>
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
  prefetch_ready_workers_(false),
  finished_(false),
  prefetch_queue_depth_(spec.GetArgument<int>("prefeth_queue_depth")),
  prefetched_batch_queue_(prefetch_queue_depth_),
  curr_batch_consumer_(0),
  curr_batch_producer_(0),
  consumer_cycle_(false),
  producer_cycle_(false),
  samples_processed_(0),
  batch_stop_(false),
  prefetch_error_(nullptr) {}

  ~DataReader() noexcept override {
    StopPrefetchThread();
    for (auto p_b : prefetched_batch_queue_) {
      for (size_t i = 0; i < p_b.size(); ++i) {
        // return unconsumed batches
        if (p_b[i]) {
          loader_->ReturnTensor(p_b[i]);
        }
      }
    }
  }

  // perform the prefetching operation
  virtual void Prefetch() {
    // We actually prepare the next batch
    TimeRange tr("Prefetching for " + to_string(curr_batch_producer_), TimeRange::kRed);
    prefetched_batch_queue_[curr_batch_producer_].reserve(Operator<Backend>::batch_size_);
    prefetched_batch_queue_[curr_batch_producer_].clear();

    for (int i = 0; i < Operator<Backend>::batch_size_; ++i) {
      auto* t = loader_->ReadOne();
      prefetched_batch_queue_[curr_batch_producer_].push_back(t);
    }
  }

  // Main prefetch work loop
  void PrefetchWorker() {
    {
      std::unique_lock<std::mutex> lock(prefetch_access_mutex_);

      // if a result is already ready, wait until it's consumed
      producer_.wait(lock, [&]() { return !IsFull(); });
    }
    while (!finished_) {
      try {
        Prefetch();
      } catch (const std::exception& e) {
        prefetch_error_.reset(new std::string(e.what()));
        finished_ = true;
        consumer_.notify_all();
        return;
      }

      {
        std::unique_lock<std::mutex> lock(prefetch_access_mutex_);
        // notify the consumer of a result ready to consume
        curr_batch_producer_ = NextIdx(curr_batch_producer_, producer_cycle_);

        consumer_.notify_all();

        // wait until the result is consumed
        producer_.wait(lock, [&]() { return finished_ || !IsFull(); });
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
        consumer_.wait(lock, [this]() { return finished_ || !IsEmpty(); });
        finished_ = true;
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

  // CPUBackend operators
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
        TimeRange tr("CONSUMER " + to_string(curr_batch_consumer_) +  " waiting",
                     TimeRange::kMagenta);
        std::unique_lock<std::mutex> prefetch_lock(prefetch_access_mutex_);

        // Wait until prefetch is ready
        consumer_.wait(prefetch_lock, [this]() { return finished_ || !IsEmpty(); });

        if (prefetch_error_) {
          DALI_FAIL("Prefetching failed: " + dali::string(*prefetch_error_));
        }

        // signal the other workers we're ready
        prefetch_ready_workers_ = true;
      }
    }

    TimeRange tr("Consuming " + to_string(curr_batch_consumer_), TimeRange::kViolet);

    // consume batch
    Operator<Backend>::Run(ws);

    loader_->ReturnTensor(prefetched_batch_queue_[curr_batch_consumer_][ws->data_idx()]);
    prefetched_batch_queue_[curr_batch_consumer_][ws->data_idx()] = nullptr;

    samples_processed_++;

    // lock, check if batch is finished, notify
    {
      std::unique_lock<std::mutex> lock(prefetch_access_mutex_);

      // if we need to stop, stop.
      if (batch_stop_) return;

      // if we've consumed all samples in this batch, reset state and stop
      if (samples_processed_.load() == Operator<Backend>::batch_size_) {
        prefetch_ready_workers_ = false;
        curr_batch_consumer_ = NextIdx(curr_batch_consumer_, consumer_cycle_);

        producer_.notify_one();
        samples_processed_ = 0;
        batch_stop_ = true;
      }
      return;
    }
  }

  // GPUBackend operators
  void Run(DeviceWorkspace* ws) override {
    StartPrefetchThread();

    // grab the actual prefetching lock
    std::unique_lock<std::mutex> prefetch_lock(prefetch_access_mutex_);

    // Wait until prefetch is ready
    consumer_.wait(prefetch_lock, [this]() { return finished_ || !IsEmpty(); });

    if (prefetch_error_) {
      DALI_FAIL("Prefetching failed: " + dali::string(*prefetch_error_));
    }

    producer_.notify_one();

    Operator<Backend>::Run(ws);

    CUDA_CALL(cudaStreamSynchronize(ws->stream()));

    for (auto &sample : prefetched_batch_queue_[curr_batch_consumer_]) {
        loader_->ReturnTensor(sample);
    }

    curr_batch_consumer_ = NextIdx(curr_batch_consumer_, consumer_cycle_);
    producer_.notify_one();
  }


  Index epoch_size() const override {
    return loader_->Size();
  }


  LoadTarget* GetSample(int sample_idx) {
    return prefetched_batch_queue_[curr_batch_consumer_][sample_idx];
  }

 protected:
  int NextIdx(int curr_batch, bool& cycle) {
    if (curr_batch == prefetch_queue_depth_ - 1)
      cycle = !cycle;
    return (curr_batch + 1) % prefetch_queue_depth_;
  }

  bool IsEmpty() {
    return curr_batch_producer_ == curr_batch_consumer_
           && consumer_cycle_ == producer_cycle_;
  }

  bool IsFull() {
    return curr_batch_producer_ == curr_batch_consumer_
           && consumer_cycle_ != producer_cycle_;
  }

  std::unique_ptr<std::thread> prefetch_thread_;

  // mutex to control access to the producer
  std::mutex prefetch_access_mutex_;
  std::mutex worker_mutex_;

  // signals for producer and consumer
  std::condition_variable producer_, consumer_;
  std::condition_variable worker_threads_;

  // signal that a complete batch has been prefetched
  std::atomic<bool> prefetch_ready_workers_;

  // signal that the prefetch thread has finished
  std::atomic<bool> finished_;

  // prefetched batch
  int prefetch_queue_depth_;
  std::vector<std::vector<LoadTarget*>> prefetched_batch_queue_;
  int curr_batch_consumer_;
  int curr_batch_producer_;
  bool consumer_cycle_;
  bool producer_cycle_;

  // keep track of how many samples have been processed
  // over all threads.
  std::atomic<int> samples_processed_;

  // notify threads to stop processing
  std::atomic<bool> batch_stop_;

  // set to error string when prefetch worker fails
  std::unique_ptr<std::string> prefetch_error_;

  // Loader
  std::unique_ptr<Loader<Backend, LoadTarget>> loader_;

  // Parser
  std::unique_ptr<Parser<ParseTarget>> parser_;
};

#define USE_READER_OPERATOR_MEMBERS_1(Backend, LoadTarget) \
  using DataReader<Backend, LoadTarget>::loader_;          \
  using DataReader<Backend, LoadTarget>::parser_;          \
  using DataReader<Backend, LoadTarget>::prefetched_batch_queue_;

#define USE_READER_OPERATOR_MEMBERS_2(Backend, LoadTarget, ParseTarget) \
  using DataReader<Backend, LoadTarget, ParseTarget>::loader_;          \
  using DataReader<Backend, LoadTarget, ParseTarget>::parser_;          \
  using DataReader<Backend, LoadTarget, ParseTarget>::prefetched_batch_queue_;

#define USE_READER_OPERATOR_MEMBERS(Backend, ...) \
  GET_MACRO(__VA_ARGS__,                          \
            USE_READER_OPERATOR_MEMBERS_2,        \
            USE_READER_OPERATOR_MEMBERS_1)(Backend, __VA_ARGS__)

};  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_READER_OP_H_
