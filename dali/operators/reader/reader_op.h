// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_READER_OP_H_
#define DALI_OPERATORS_READER_READER_OP_H_

#include <atomic>
#include <condition_variable>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <unordered_map>

#include "dali/core/nvtx.h"
#include "dali/operators/reader/loader/loader.h"
#include "dali/operators/reader/parser/parser.h"
#include "dali/pipeline/operator/operator.h"

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
  using LoadTargetPtr = std::shared_ptr<LoadTarget>;

  inline explicit DataReader(const OpSpec& spec)
      : Operator<Backend>(spec),
        finished_(false),
        prefetch_queue_depth_(spec.GetArgument<int>("prefetch_queue_depth")),
        skip_cached_images_(spec.GetArgument<bool>("skip_cached_images")),
        prefetched_batch_queue_(prefetch_queue_depth_),
        curr_batch_consumer_(0),
        curr_batch_producer_(0),
        consumer_cycle_(false),
        producer_cycle_(false),
        device_id_(-1),
        samples_processed_(0) {
          if (std::is_same<Backend, GPUBackend>::value) {
            device_id_ = spec.GetArgument<int>("device_id");
          }
        }

  ~DataReader() noexcept override {
    StopPrefetchThread();
    for (auto &batch : prefetched_batch_queue_) {
      // make share_ptr do their job while loader is still alive
      // and RecycleTensor could be safely executed
      batch.clear();
    }
  }

  // perform the prefetching operation
  virtual void Prefetch() {
    // We actually prepare the next batch
    DomainTimeRange tr("[DALI][DataReader] Prefetch #" + to_string(curr_batch_producer_),
                       DomainTimeRange::kRed);
    auto &curr_batch = prefetched_batch_queue_[curr_batch_producer_];
    curr_batch.reserve(max_batch_size_);
    curr_batch.clear();
    for (int i = 0; i < max_batch_size_; ++i) {
      curr_batch.push_back(loader_->ReadOne(i == 0));
    }
  }

  // Main prefetch work loop
  void PrefetchWorker() {
    DeviceGuard g(device_id_);
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

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    // If necessary start prefetching thread and wait for a consumable batch
    StartPrefetchThread();
    ConsumerWait();
    return false;
  }

  using Operator<Backend>::Run;

  // CPUBackend operators
  void Run(HostWorkspace &ws) override {
    // consume batch
    DomainTimeRange tr("[DALI][DataReader] Run #" + to_string(curr_batch_consumer_),
                       DomainTimeRange::kViolet);

    // This is synchronous call for CPU Backend
    Operator<Backend>::Run(ws);

    EnforceUniformOutput(ws);

    // Notify that we have consumed whole batch
    ConsumerAdvanceQueue();
  }

  void EnforceUniformOutput(const HostWorkspace &ws) const {
    for (int out_idx = 0; out_idx < ws.NumOutput(); out_idx++) {
      auto &out = ws.OutputRef<CPUBackend>(out_idx);
      int n = out.ntensor();
      if (n < 2)
        continue;
      auto type0 = out[0].type().id();
      int ndim0 = out[0].shape().size();
      for (int i = 1; i < n; i++) {
        auto type = out[i].type().id();
        DALI_ENFORCE(type == type0, make_string("Inconsistent data! "
        "The data produced by the reader has inconsistent type:\n"
        "type of outputs[", out_idx, "][", i, "] is ", type, " whereas\n"
        "type of outputs[", out_idx, "][0] is ", type0));

        int ndim = out[i].shape().size();
        DALI_ENFORCE(ndim == ndim0, make_string("Inconsistent data! "
        "The data produced by the reader has inconsistent dimensionality:\n"
        "outputs[", out_idx, "][", i, "] has ", ndim, " dimensions whereas\n"
        "outputs[", out_idx, "][0] has ", ndim0, " dimensions."));      }
    }
  }


  // GPUBackend operators
  void Run(DeviceWorkspace &ws) override {
    // Consume batch
    Operator<Backend>::Run(ws);
    CUDA_CALL(cudaStreamSynchronize(ws.stream()));
    for (int sample_idx = 0; sample_idx < max_batch_size_; sample_idx++) {
      auto sample = MoveSample(sample_idx);
    }

    // Notify we have consumed a batch
    ConsumerAdvanceQueue();
  }

  ReaderMeta GetReaderMeta() const override {
    ReaderMeta ret;
    ret.epoch_size = loader_->Size(false);
    ret.epoch_size_padded = loader_->Size(true);
    ret.number_of_shards = loader_->GetNumShards();
    ret.shard_id = loader_->GetShardId();
    ret.pad_last_batch = loader_->PadLastBatch();
    ret.stick_to_shard = loader_->StickToShard();
    return ret;
  }

  inline std::vector<std::shared_ptr<LoadTarget>>& GetCurrBatch() {
    return prefetched_batch_queue_[curr_batch_consumer_];
  }

  inline const std::vector<std::shared_ptr<LoadTarget>>& GetCurrBatch() const {
    return prefetched_batch_queue_[curr_batch_consumer_];
  }

  inline int GetCurrBatchSize() const {
    return GetCurrBatch().size();
  }

  inline LoadTarget& GetSample(int sample_idx) {
    return *GetCurrBatch()[sample_idx];
  }

  LoadTargetPtr MoveSample(int sample_idx) {
    auto &sample = prefetched_batch_queue_[curr_batch_consumer_][sample_idx];
    auto sample_ptr = std::move(sample);
    sample = {};
    return sample_ptr;
  }

 protected:
  void ParseIfNeeded(const Tensor<CPUBackend>& tensor, SampleWorkspace* ws) {
    using OutputCache = std::unordered_map<std::string, std::vector<Tensor<CPUBackend>>>;
    static OutputCache output_cache;
    static std::mutex output_cache_mutex;

    const auto& source_info = tensor.GetSourceInfo();
    const auto should_skip_sample = tensor.ShouldSkipSample();
    const std::size_t num_outputs = ws->NumOutput();

    if (should_skip_sample) {
      std::lock_guard<std::mutex> lock(output_cache_mutex);
      auto it = output_cache.find(source_info);
      DALI_ENFORCE(it != output_cache.end(),
        "Image `" + source_info + "` should be in cache (cache size: "
        + std::to_string(output_cache.size()) + ")");
      auto& cached_outputs = it->second;
      DALI_ENFORCE(cached_outputs.size() == num_outputs,
        "Unexpected number of outputs");
      for (std::size_t i = 0; i < cached_outputs.size(); i++) {
        auto& output = ws->Output<CPUBackend>(i);
        output.Copy(cached_outputs[i], 0);
      }
      return;
    }

    parser_->Parse(tensor, ws);

    if (skip_cached_images_) {
      std::lock_guard<std::mutex> lock(output_cache_mutex);
      if (output_cache.find(source_info) != output_cache.end()) {
        return;
      }

      auto& cached_outputs = output_cache[source_info];
      cached_outputs.resize(num_outputs);

      // We don't want to cache the image itself
      auto& first_output = cached_outputs[0];
      first_output.set_pinned(false);
      first_output.SetSourceInfo(source_info);
      first_output.SetSkipSample(should_skip_sample);
      first_output.set_type(TypeInfo::Create<uint8_t>());
      first_output.Resize({1});

      for (std::size_t i = 1; i < cached_outputs.size(); i++) {
        auto& output = ws->Output<CPUBackend>(i);
        cached_outputs[i].set_pinned(false);
        cached_outputs[i].Copy(output, 0);
      }
    }
  }

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
    DomainTimeRange tr("[DALI][DataReader] ConsumerWait #" + to_string(curr_batch_consumer_),
                 DomainTimeRange::kMagenta);
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
    return curr_batch_producer_ == curr_batch_consumer_ && consumer_cycle_ == producer_cycle_;
  }

  bool IsPrefetchQueueFull() {
    return curr_batch_producer_ == curr_batch_consumer_ && consumer_cycle_ != producer_cycle_;
  }

  USE_OPERATOR_MEMBERS();

  std::thread prefetch_thread_;

  // mutex to control access to the producer
  std::mutex prefetch_access_mutex_;

  // signals for producer and consumer
  std::condition_variable producer_, consumer_;

  // signal that the prefetch thread has finished
  std::atomic<bool> finished_;

  // prefetched batch
  int prefetch_queue_depth_;
  bool skip_cached_images_;
  using BatchQueueElement = std::vector<LoadTargetPtr>;
  std::vector<BatchQueueElement> prefetched_batch_queue_;
  int curr_batch_consumer_;
  int curr_batch_producer_;
  bool consumer_cycle_;
  bool producer_cycle_;
  int device_id_;

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

#endif  // DALI_OPERATORS_READER_READER_OP_H_
