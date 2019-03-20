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
#include <utility>
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
  using LoadTargetPtr = std::unique_ptr<LoadTarget>;

  inline explicit DataReader(const OpSpec& spec)
      : Operator<Backend>(spec),
        finished_(false),
        prefetch_queue_depth_(spec.GetArgument<int>("prefetch_queue_depth")),
        prefetched_batch_queue_(prefetch_queue_depth_),
        curr_batch_consumer_(0),
        curr_batch_producer_(0),
        consumer_cycle_(false),
        producer_cycle_(false),
        samples_processed_(0) {}

  ~DataReader() noexcept override {
    StopPrefetchThread();
    for (auto &batch : prefetched_batch_queue_) {
      for (auto &sample : batch) {
        if (sample)
          loader_->RecycleTensor(std::move(sample));
      }
    }
  }

  // perform the prefetching operation
  virtual void Prefetch() {
    // We actually prepare the next batch
    TimeRange tr("DataReader::Prefetch #" + to_string(curr_batch_producer_), TimeRange::kRed);
    auto &curr_batch = prefetched_batch_queue_[curr_batch_producer_];
    curr_batch.reserve(Operator<Backend>::batch_size_);
    curr_batch.clear();
    for (int i = 0; i < Operator<Backend>::batch_size_; ++i) {
      curr_batch.push_back(loader_->ReadOne());
    }
  }

  // Main prefetch work loop
  void PrefetchWorker() {
    ProducerWait();
    while (!finished_) {
      try {
        Prefetch();
      } catch (const std::exception& e) {
        ProducerStop(std::current_exception());
        return;
      }
      ProducerAdvanceQueue();
      ProducerWait();
    }
  }

  // to be called in constructor
  void StartPrefetchThread() {
    std::lock_guard<std::mutex> lock(prefetch_access_mutex_);
    // if thread hasn't been started yet, start it
    if (prefetch_thread_.joinable()) return;
    prefetch_thread_ = std::thread(&DataReader::PrefetchWorker, this);
  }

  // to be called in destructor
  void StopPrefetchThread() {
    ProducerStop();
    if (prefetch_thread_.joinable()) {
      producer_.notify_one();
      // join the prefetch thread and destroy it
      prefetch_thread_.join();
      prefetch_thread_ = {};
    }
  }

  // CPUBackend operators
  void Run(SampleWorkspace* ws) override {
    // If necessary start prefetching thread and wait for a consumable batch
    StartPrefetchThread();
    ConsumerWait();

    // consume sample
    TimeRange tr("DataReader::Run #" + to_string(curr_batch_consumer_), TimeRange::kViolet);
    Operator<Backend>::Run(ws);
    const auto data_idx = ws->data_idx();
    loader_->RecycleTensor(MoveSample(data_idx));
    auto curr_sample_id = samples_processed_.fetch_add(1);
    // if we've processed the whole batch, notify it
    if (curr_sample_id == Operator<Backend>::batch_size_ - 1) {
      samples_processed_ = 0;
      ConsumerAdvanceQueue();
    }
  }

  // GPUBackend operators
  void Run(DeviceWorkspace* ws) override {
    // If necessary start prefetching thread and wait for a consumable batch
    StartPrefetchThread();
    ConsumerWait();

    // Consume batch
    Operator<Backend>::Run(ws);
    CUDA_CALL(cudaStreamSynchronize(ws->stream()));
    for (int sample_idx = 0; sample_idx < Operator<Backend>::batch_size_; sample_idx++) {
      loader_->RecycleTensor(MoveSample(sample_idx));
    }

    // Notify we have consumed a batch
    ConsumerAdvanceQueue();
  }

  Index epoch_size() const override {
    return loader_->Size();
  }

  LoadTarget& GetSample(int sample_idx) {
    return *prefetched_batch_queue_[curr_batch_consumer_][sample_idx];
  }

  LoadTargetPtr MoveSample(int sample_idx) {
    auto &sample = prefetched_batch_queue_[curr_batch_consumer_][sample_idx];
    auto sample_ptr = std::move(sample);
    sample = {};
    return sample_ptr;
  }

 protected:
  void ProducerStop(std::exception_ptr error = nullptr) {
    {
      std::lock_guard<std::mutex> lock(prefetch_access_mutex_);
      finished_ = true;
      if (error)
        prefetch_error_ = error;
    }
    consumer_.notify_all();
  }

  void ProducerAdvanceQueue() {
    {
      std::lock_guard<std::mutex> lock(prefetch_access_mutex_);
      AdvanceIndex(curr_batch_producer_, producer_cycle_);
    }
    consumer_.notify_all();
  }

  void ProducerWait() {
    std::unique_lock<std::mutex> lock(prefetch_access_mutex_);
    producer_.wait(lock, [&]() { return finished_ || !IsPrefetchQueueFull(); });
  }

  void ConsumerWait() {
    TimeRange tr("DataReader::ConsumerWait #" + to_string(curr_batch_consumer_),
                 TimeRange::kMagenta);
    std::unique_lock<std::mutex> prefetch_lock(prefetch_access_mutex_);
    consumer_.wait(prefetch_lock, [this]() { return finished_ || !IsPrefetchQueueEmpty(); });
    if (prefetch_error_) std::rethrow_exception(prefetch_error_);
  }

  void ConsumerAdvanceQueue() {
    {
      std::lock_guard<std::mutex> lock(prefetch_access_mutex_);
      AdvanceIndex(curr_batch_consumer_, consumer_cycle_);
    }
    producer_.notify_one();
  }

  void AdvanceIndex(int& index, bool& cycle) {
    index = (index + 1) % prefetch_queue_depth_;
    if (index == 0) cycle = !cycle;
  }

  bool IsPrefetchQueueEmpty() {
    return curr_batch_producer_ == curr_batch_consumer_
           && consumer_cycle_ == producer_cycle_;
  }

  bool IsPrefetchQueueFull() {
    return curr_batch_producer_ == curr_batch_consumer_
           && consumer_cycle_ != producer_cycle_;
  }

  std::thread prefetch_thread_;

  // mutex to control access to the producer
  std::mutex prefetch_access_mutex_;

  // signals for producer and consumer
  std::condition_variable producer_, consumer_;

  // signal that the prefetch thread has finished
  std::atomic<bool> finished_;

  // prefetched batch
  int prefetch_queue_depth_;
  using BatchQueueElement = std::vector<LoadTargetPtr>;
  std::vector<BatchQueueElement> prefetched_batch_queue_;
  int curr_batch_consumer_;
  int curr_batch_producer_;
  bool consumer_cycle_;
  bool producer_cycle_;

  // keep track of how many samples have been processed over all threads.
  std::atomic<int> samples_processed_;

  // stores any catched exceptions in the prefetch worker
  std::exception_ptr prefetch_error_;

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
