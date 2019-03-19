// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_READER_LOADER_LOADER_H_
#define DALI_PIPELINE_OPERATORS_READER_LOADER_LOADER_H_

#include <map>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {

DLL_PUBLIC size_t start_index(const size_t shard_id,
                              const size_t shard_num,
                              const size_t size);
/**
 * @brief Base class for Loaders, responsible for reading samples from resource of some kind
 *        into memory.
 *
 * @tparam Backend
 * @tparam LoadTarget Type into which samples are loaded.
 */
template <typename Backend, typename LoadTarget>
class Loader {
 public:
  using LoadTarget_t = LoadTarget;
  explicit Loader(const OpSpec& options)
    : shuffle_(options.GetArgument<bool>("random_shuffle")),
      initial_buffer_fill_(shuffle_ ? options.GetArgument<int>("initial_fill") : 1),
      initial_empty_size_(2 * options.GetArgument<int>("prefetch_queue_depth")
                          * options.GetArgument<int>("batch_size")),
      tensor_init_bytes_(options.GetArgument<int>("tensor_init_bytes")),
      seed_(options.GetArgument<Index>("seed")),
      shard_id_(options.GetArgument<int>("shard_id")),
      num_shards_(options.GetArgument<int>("num_shards")),
      read_ahead_(options.GetArgument<bool>("read_ahead")),
      stick_to_shard_(options.GetArgument<bool>("stick_to_shard")) {
    DALI_ENFORCE(initial_empty_size_ > 0, "Batch size needs to be greater than 0");
    DALI_ENFORCE(num_shards_ > shard_id_, "num_shards needs to be greater than shard_id");
    // initialize a random distribution -- this will be
    // used to pick from our sample buffer
    dis = std::uniform_int_distribution<>(0, initial_buffer_fill_);
    std::seed_seq seq({seed_});
    e_ = std::default_random_engine(seq);
  }

  virtual ~Loader() {
    // delete all the temporary tensors
    while (!sample_buffer_.empty()) {
      LoadTarget * t = sample_buffer_.back();
      delete t;
      sample_buffer_.pop_back();
    }
    while (!empty_tensors_.empty()) {
      LoadTarget * t = empty_tensors_.back();
      delete t;
      empty_tensors_.pop_back();
    }
  }

  virtual void PrepareEmpty(LoadTarget *tensor) {
    PrepareEmptyTensor(tensor);
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, Tensor<CPUBackend>>::value>::type
  PrepareEmptyTensor(T *tensor) {
    tensor->set_pinned(false);
    // Initialize tensors to a set size to limit expensive reallocations
    tensor->Resize({tensor_init_bytes_});
    tensor->template mutable_data<uint8_t>();
  }

  template <typename T>
  typename std::enable_if<!std::is_same<T, Tensor<CPUBackend>>::value>::type
  PrepareEmptyTensor(T *) {
    constexpr bool T_is_Tensor = std::is_same<T, Tensor<CPUBackend>>::value;
    DALI_ENFORCE(T_is_Tensor,
      "Please overload PrepareEmpty for custom LoadTarget type other than Tensor");
  }


  // Get a random read sample
  LoadTarget* ReadOne() {
    TimeRange tr("[Loader] ReadOne", TimeRange::kGreen1);
    // perform an iniital buffer fill if it hasn't already happened
    if (!initial_buffer_filled_) {
      TimeRange tr("[Loader] Filling initial buffer", TimeRange::kBlue1);
      std::lock_guard<std::mutex> lock(return_mutex_);
      // Read an initial number of samples to fill our
      // sample buffer
      for (int i = 0; i < initial_buffer_fill_; ++i) {
        LoadTarget* tensor = new LoadTarget();
        PrepareEmpty(tensor);

        ReadSample(tensor);
        sample_buffer_.push_back(tensor);
      }

      TimeRange tr2("[Loader] Filling empty list", TimeRange::kOrange);
      // need some entries in the empty_tensors_ list
      for (int i = 0; i < initial_empty_size_; ++i) {
        LoadTarget* tensor = new LoadTarget();
        PrepareEmpty(tensor);
        empty_tensors_.push_back(tensor);
      }

      initial_buffer_filled_ = true;
    }
    // choose the random index
    int idx = shuffle_ ? dis(e_) % sample_buffer_.size() : 0;

    // swap end and idx, return the tensor to empties
    std::swap(sample_buffer_[idx], sample_buffer_.back());
    // remove last element
    LoadTarget* elem = sample_buffer_.back();
    sample_buffer_.pop_back();

    // now grab an empty tensor, fill it and add to filled buffers
    // empty_tensors_ needs to be thread-safe w.r.t. RecycleTensor()
    // being called by multiple consumer threads
    LoadTarget* t;
    {
      std::lock_guard<std::mutex> lock(return_mutex_);
      DALI_ENFORCE(empty_tensors_.size() > 0, "No empty tensors - did you forget to return them?");
      t = empty_tensors_.back();
      empty_tensors_.pop_back();
    }
    ReadSample(t);
    sample_buffer_.push_back(t);

    return elem;
  }

  // return a tensor to the empty pile
  // called by multiple consumer threads
  void RecycleTensor(LoadTarget* tensor) {
    std::lock_guard<std::mutex> lock(return_mutex_);
    const auto it = std::find(empty_tensors_.begin(), empty_tensors_.end(), tensor);
    DALI_ENFORCE(it == empty_tensors_.end(),
      "Tensor " + std::to_string(reinterpret_cast<uint64_t>(tensor)) +
          " is already in empty_tensors_");
    empty_tensors_.push_back(tensor);
  }

  // Read an actual sample from the FileStore,
  // used to populate the sample buffer for "shuffled"
  // reads.
  virtual void ReadSample(LoadTarget* tensor) = 0;

  // Give the size of the data accessed through the Loader
  virtual Index Size() = 0;

 protected:
  virtual void MoveToNextShard(Index current_index) {
    if (IsNextShard(current_index)) {
      Reset(stick_to_shard_);
    }
  }
  // Reset reader to the first sample
  virtual void Reset(bool wrap_to_shard) = 0;

  // Check if given reader moved to the next shard
  virtual inline bool IsNextShard(Index current_index) {
     return current_index >= Size() ||
            (stick_to_shard_ && shard_id_ + 1 < num_shards_ &&
            current_index >= static_cast<Index>(start_index(shard_id_ + 1, num_shards_, Size())));
  }
  std::vector<LoadTarget*> sample_buffer_;

  std::vector<LoadTarget*> empty_tensors_;

  // number of samples to initialize buffer with
  // ~1 minibatch seems reasonable
  bool shuffle_;
  const int initial_buffer_fill_;
  const int initial_empty_size_;
  const int tensor_init_bytes_;
  bool initial_buffer_filled_ = false;

  // rng
  std::default_random_engine e_;
  std::uniform_int_distribution<> dis;
  Index seed_;

  // control return of tensors
  std::mutex return_mutex_;

  // sharding
  const int shard_id_;
  const int num_shards_;

  // if read data need to be copied or can be just shared with tensor
  bool copy_read_data_;
  // if accessed files should be loaded into memory in advance at the first access
  bool read_ahead_;
  // if reader for the given GPU should read over and over the same shard or should go through
  // whole data set
  bool stick_to_shard_;
};

};  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_LOADER_LOADER_H_
