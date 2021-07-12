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

#ifndef DALI_OPERATORS_READER_LOADER_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_LOADER_H_

#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <deque>
#include <atomic>

#include "dali/core/nvtx.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/operators/decoder/cache/image_cache_factory.h"

namespace dali {

DLL_PUBLIC size_t start_index(const size_t shard_id,
                              const size_t shard_num,
                              const size_t size);

DLL_PUBLIC Index num_samples(const size_t shard_num,
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
  using LoadTargetUniquePtr = std::unique_ptr<LoadTarget>;
  using LoadTargetSharedPtr = std::shared_ptr<LoadTarget>;
  explicit Loader(const OpSpec& options)
    : shuffle_(options.GetArgument<bool>("random_shuffle")),
      initial_buffer_fill_(shuffle_ ? options.GetArgument<int>("initial_fill") : 1),
      initial_empty_size_(2 * options.GetArgument<int>("prefetch_queue_depth")
                          * options.GetArgument<int>("max_batch_size")),
      tensor_init_bytes_(options.GetArgument<int>("tensor_init_bytes")),
      seed_(options.GetArgument<Index>("seed")),
      shard_id_(options.GetArgument<int>("shard_id")),
      num_shards_(options.GetArgument<int>("num_shards")),
      copy_read_data_(false),
      read_ahead_(options.GetArgument<bool>("read_ahead")),
      stick_to_shard_(options.GetArgument<bool>("stick_to_shard")),
      device_id_(options.GetArgument<int>("device_id")),
      skip_cached_images_(options.GetArgument<bool>("skip_cached_images")),
      lazy_init_(options.GetArgument<bool>("lazy_init")),
      loading_flag_(false),
      read_sample_counter_(0),
      returned_sample_counter_(0),
      pad_last_batch_(options.GetArgument<bool>("pad_last_batch")),
      dont_use_mmap_(options.GetArgument<bool>("dont_use_mmap")) {
    DALI_ENFORCE(initial_empty_size_ > 0, "Batch size needs to be greater than 0");
    DALI_ENFORCE(num_shards_ > shard_id_, "num_shards needs to be greater than shard_id");
    // initialize a random distribution -- this will be
    // used to pick from our sample buffer
    std::seed_seq seq({seed_});
    e_ = std::default_random_engine(seq);
    virtual_shard_id_ = shard_id_;
  }

  virtual ~Loader() {
    sample_buffer_.clear();
    empty_tensors_.clear();
  }

  // We need this two stage init because overriden PrepareMetadata
  // is not known in Loader ctor
  void Init() {
    if (!lazy_init_) {
      PrepareMetadata();
    }
  }

  virtual void PrepareEmpty(LoadTarget& tensor) {
    PrepareEmptyTensor(tensor);
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, Tensor<GPUBackend>>::value ||
                   std::is_same<T, Tensor<CPUBackend>>::value>
  PrepareEmptyTensor(T& tensor) {
    tensor.set_pinned(false);
    // Initialize tensors to a set size to limit expensive reallocations
    tensor.Resize({tensor_init_bytes_});
    tensor.template mutable_data<uint8_t>();
  }

  template <typename T>
    std::enable_if_t<!(std::is_same<T, Tensor<CPUBackend>>::value ||
                       std::is_same<T, Tensor<GPUBackend>>::value)>
  PrepareEmptyTensor(T&) {
    DALI_ERROR("Please overload PrepareEmpty for custom LoadTarget type other than Tensor");
  }

  // Get a random read sample
  LoadTargetSharedPtr ReadOne(bool is_new_batch) {
    PrepareMetadata();
    DomainTimeRange tr("[DALI][Loader] ReadOne", DomainTimeRange::kGreen1);
    // perform an initial buffer fill if it hasn't already happened
    if (!initial_buffer_filled_) {
      DomainTimeRange tr("[DALI][Loader] Filling initial buffer", DomainTimeRange::kBlue1);
      shards_.push_back({0, 0});

      // Read an initial number of samples to fill our
      // sample buffer
      for (int i = 0; i < initial_buffer_fill_; ++i) {
        auto tensor_ptr = LoadTargetUniquePtr(new LoadTarget());
        PrepareEmpty(*tensor_ptr);
        ReadSample(*tensor_ptr);
        IncreaseReadSampleCounter();
        sample_buffer_.push_back(std::move(tensor_ptr));
        ++shards_.back().end;
      }

      // need some entries in the empty_tensors_ list
      DomainTimeRange tr2("[DALI][Loader] Filling empty list", DomainTimeRange::kOrange);
      std::lock_guard<std::mutex> lock(empty_tensors_mutex_);
      for (int i = 0; i < initial_empty_size_; ++i) {
        auto tensor_ptr = LoadTargetUniquePtr(new LoadTarget());
        PrepareEmpty(*tensor_ptr);
        empty_tensors_.push_back(std::move(tensor_ptr));
      }

      initial_buffer_filled_ = true;
    }

    if (shards_.front().start == shards_.front().end) {
      // If the reader has depleted samples from the given shard, but shards are not equal
      // and we need to pad samples inside batch (even create a whole new dummy batch) using padding
      // just to return in each shard the same number of samples and batches within the epoch.
      // It happened only when pad_last_batch_ is set
      // First part of this condition makes sure that the same number of batches is returned in each
      // shard. Second makes sure that padding is done up to the full batch. For the first sample in
      // the batch is_new_batch is set so it means that padding may be no longer needed
      if ((returned_sample_counter_  < num_samples(num_shards_, Size()) || !is_new_batch) &&
        pad_last_batch_) {
        ++returned_sample_counter_;
        return last_sample_ptr_tmp;
      }
      // remove shard that was fully consumed
      shards_.pop_front();
      returned_sample_counter_ = 0;
    }

    // choose the random index
    std::uniform_int_distribution<> dis;
    dis = std::uniform_int_distribution<>(0, shards_.front().end - shards_.front().start - 1);

    int offset = shuffle_ ? dis(e_) : 0;
    Index idx = (shards_.front().start + offset) % sample_buffer_.size();
    LoadTargetSharedPtr sample_ptr(sample_buffer_[idx].release(),
      [this](LoadTarget* sample) {
        LoadTargetUniquePtr recycle_ptr(sample);
        RecycleTensor(std::move(recycle_ptr));
    });
    std::swap(sample_buffer_[idx], sample_buffer_[shards_.front().start % sample_buffer_.size()]);
    // now grab an empty tensor, fill it and add to filled buffers
    // empty_tensors_ needs to be thread-safe w.r.t. RecycleTensor()
    // being called by multiple consumer threads
    LoadTargetUniquePtr tensor_ptr;
    {
      std::lock_guard<std::mutex> lock(empty_tensors_mutex_);
      DALI_ENFORCE(empty_tensors_.size() > 0, "No empty tensors - did you forget to return them?");
      tensor_ptr = std::move(empty_tensors_.back());
      empty_tensors_.pop_back();
    }
    ReadSample(*tensor_ptr);
    IncreaseReadSampleCounter();
    std::swap(sample_buffer_[shards_.back().end % sample_buffer_.size()], tensor_ptr);
    ++shards_.back().end;
    last_sample_ptr_tmp = sample_ptr;

    shards_.front().start++;
    returned_sample_counter_++;
    return sample_ptr;
  }

  // return a tensor to the empty pile
  // called by multiple consumer threads
  void RecycleTensor(LoadTargetUniquePtr&& tensor_ptr) {
    std::lock_guard<std::mutex> lock(empty_tensors_mutex_);
    empty_tensors_.push_back(std::move(tensor_ptr));
  }

  // Read an actual sample from the FileStore,
  // used to populate the sample buffer for "shuffled"
  // reads.
  virtual void ReadSample(LoadTarget& tensor) = 0;

  void PrepareMetadata() {
    if (!loading_flag_) {
      std::lock_guard<std::mutex> l(prepare_metadata_mutex_);
      if (!loading_flag_) {
        PrepareMetadataImpl();
        std::atomic_thread_fence(std::memory_order_release);
        loading_flag_ = true;
      }
    }
  }

  // Give the size of the data accessed through the Loader
  Index Size(bool consider_padding = false) {
    PrepareMetadata();
    if (pad_last_batch_ && consider_padding) {
      return num_samples(num_shards_, SizeImpl()) * num_shards_;
    } else {
      return SizeImpl();
    }
  }

  int GetNumShards() {
    return num_shards_;
  }

  int GetShardId() {
    return shard_id_;
  }

  int PadLastBatch() {
    return pad_last_batch_;
  }

  int StickToShard() {
    return stick_to_shard_;
  }

 protected:
  virtual Index SizeImpl() = 0;

  virtual void PrepareMetadataImpl() {}

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

  inline bool IsNextShardRelative(Index already_read, int virtual_shard_id) {
     Index current_index = already_read
                         + static_cast<Index>(start_index(virtual_shard_id, num_shards_, Size()));
     return current_index >= Size() ||
            (virtual_shard_id + 1 < num_shards_ &&
              current_index >=
              static_cast<Index>(start_index(virtual_shard_id + 1, num_shards_, Size())));
  }

  inline void IncreaseReadSampleCounter() {
    ++read_sample_counter_;
    if (IsNextShardRelative(read_sample_counter_ - 1, virtual_shard_id_)) {
      if (!stick_to_shard_) {
        ++virtual_shard_id_;
      }
      read_sample_counter_ = 1;
      if (virtual_shard_id_ == num_shards_) {
        virtual_shard_id_ = 0;
      }
      Index curr_elms = shards_.back().end;
      shards_.push_back({curr_elms, curr_elms});
    }
  }

  bool ShouldSkipImage(const ImageCache::ImageKey& key) {
    if (!skip_cached_images_)
      return false;

    // Fetch image cache factory only the first time that we try to load an image
    // we don't do it in construction because we are not sure that the cache was
    // created since the order of operator creation is not guaranteed.
    std::call_once(fetch_cache_, [this](){
      auto &image_cache_factory = ImageCacheFactory::Instance();
      if (image_cache_factory.IsInitialized(device_id_))
        cache_ = image_cache_factory.Get(device_id_);
    });
    return cache_ && cache_->IsCached(key);
  }

  std::vector<LoadTargetUniquePtr> sample_buffer_;

  std::vector<LoadTargetUniquePtr> empty_tensors_;

  // number of samples to initialize buffer with
  // ~1 minibatch seems reasonable
  bool shuffle_;
  const int initial_buffer_fill_;
  const int initial_empty_size_;
  const int tensor_init_bytes_;
  bool initial_buffer_filled_ = false;

  // rng
  std::default_random_engine e_;
  Index seed_;

  // control return of tensors
  std::mutex empty_tensors_mutex_;

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

  // Pipeline's device id, used to lookup if an image was cached
  int device_id_;

  // Option determining whether cached samples (at the decoder phase) should be skipped
  bool skip_cached_images_;

  // Indicate whether the dataset preparation has to be done in the constructor or during the
  // first run
  std::mutex prepare_metadata_mutex_;
  bool lazy_init_;
  bool loading_flag_;

  // Image cache
  std::once_flag fetch_cache_;
  std::shared_ptr<ImageCache> cache_;

  // Counts how many samples the reader have read already from this epoch
  Index read_sample_counter_;
  // Counts how many samples the reader have read returned in the current epoch (including padding)
  Index returned_sample_counter_;
  // If true, the last batch will be padded with the last sample so that the number
  // of samples matches batch size
  bool pad_last_batch_;
  // If true data will be always read using read function and copied instead of shared with the
  // target tensor, if false loader will try to mmap files if possible and wrap the content into
  // tensor without copy
  bool dont_use_mmap_;
  // Number of data shards that were actually read by the reader
  int virtual_shard_id_;
  // Keeps pointer to the last returned sample just in case it needs to be cloned
  LoadTargetSharedPtr last_sample_ptr_tmp;

  struct ShardBoundaries {
    Index start;
    Index end;
  };

  std::deque<ShardBoundaries> shards_;
};

template<typename T, typename... Args>
std::unique_ptr<T> InitLoader(const OpSpec& spec, Args&&... args) {
  std::unique_ptr<T> loader (new T(spec, std::forward<Args>(args)...));
  loader->Init();
  return loader;
}

};  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_LOADER_H_
