// Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @brief Structure describing Loader base state, at the begining of an epoch.
*/
struct LoaderStateSnapshot {
  std::default_random_engine rng;
  int current_epoch;
};

/**
 * @brief Base class for Loaders, responsible for reading samples from resource of some kind
 *        into memory.
 *
 * @tparam Backend
 * @tparam LoadTarget Type into which samples are loaded.
 * @tparam supports_checkpointing A marker for checkpointing support.
 */
template <typename Backend, typename LoadTarget,
          bool supports_checkpointing = false>
class Loader {
 public:
  using LoadTargetUniquePtr = std::unique_ptr<LoadTarget>;
  using LoadTargetSharedPtr = std::shared_ptr<LoadTarget>;

  struct IndexedLoadTargetSharedPtr {
    Index idx;
    LoadTargetSharedPtr ptr;
  };

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
      pad_last_batch_(options.GetArgument<bool>("pad_last_batch")),
      dont_use_mmap_(options.GetArgument<bool>("dont_use_mmap")),
      checkpointing_(options.GetArgument<bool>("checkpointing")),
      max_batch_size_(options.GetArgument<int>("max_batch_size")) {
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
    tensor.Resize({tensor_init_bytes_}, DALI_UINT8);
  }

  template <typename T>
    std::enable_if_t<!(std::is_same<T, Tensor<CPUBackend>>::value ||
                       std::is_same<T, Tensor<GPUBackend>>::value)>
  PrepareEmptyTensor(T&) {
    DALI_ERROR("Please overload PrepareEmpty for custom LoadTarget type other than Tensor");
  }

  bool IsCheckpointingEnabled() {
    return supports_checkpointing && checkpointing_;
  }

  /**
   * @brief Returns true iff the loader depleted the epoch.
   *        If true, reading the next sample for a new batch will return
   *        sample belonging the next epoch.
   */
  bool IsEpochDepleted() {
    return shards_.size() == 0 || shards_.front().start == shards_.front().end;
  }

  /**
   * @brief If called when the loader moves to a new shard (i.e. at the end of an epoch),
   *        returns the current state of the reader. Fails otherwise, as checkpointing
   *        in the middle of an epoch is not supported for now.
   */
  LoaderStateSnapshot GetStateSnapshot() {
    if constexpr (!supports_checkpointing) {
      DALI_FAIL("Checkpointing is not supported by this loader.");
    } else {
      DALI_ENFORCE(IsCheckpointingEnabled(),
                   "Checkpointing was not enabled. Please make sure you set"
                   " enable_checkpointing to True when creating the pipeline.");
      // TODO(ktokarski) Currently, the file reader can only save its
      // state at the start of an epoch
      DALI_ENFORCE(IsEpochDepleted(),
                   "Currently, checkpointing is supported only between the epochs");
      // TODO(mstaniewski): support pad_last_batch=false
      DALI_ENFORCE(pad_last_batch_,
                   "Currently, checkpointing is only supported with pad_last_batch=true");
      LoaderStateSnapshot snapshot;
      snapshot.rng = e_;
      snapshot.current_epoch = consumer_epoch_;
      SaveStateImpl(snapshot);
      return snapshot;
    }
  }

  /**
   * @brief Restores the loader's state from a snapshot.
   */
  void RestoreStateFromSnapshot(const LoaderStateSnapshot& state) {
    DALI_ENFORCE(IsCheckpointingEnabled(),
                 "Checkpointing was not enabled. Please make sure you set"
                 " enable_checkpointing to True when creating the pipeline.");
    e_ = state.rng;
    consumer_epoch_ = state.current_epoch;
    if (!stick_to_shard_)
      virtual_shard_id_ = (shard_id_ + state.current_epoch) % num_shards_;

    RestoreStateImpl(state);

    // Re-run reset
    Reset(true);
  }

  bool ShouldPadBatch(bool is_new_batch) {
    // If the reader has depleted samples from the given shard, but shards are not equal
    // and we need to pad samples inside batch (even create a whole new dummy batch) using padding
    // just to return in each shard the same number of samples and batches within the epoch.
    // It happened only when pad_last_batch_ is set
    // First part of this condition makes sure that the same number of batches is returned in each
    // shard. Second makes sure that padding is done up to the full batch. For the first sample in
    // the batch is_new_batch is set so it means that padding may be no longer needed
    return (returned_sample_counter_  < num_samples(num_shards_, Size()) || !is_new_batch) &&
            pad_last_batch_;
  }

  /**
   * @brief Fast-forwards a loader by skipping n samples.
  */
  void FastForward(Index n) {
    for (Index i = 0; i < n; i++) {
      Index pos_in_batch = (returned_sample_counter_ + i) % max_batch_size_;
      ReadOne(pos_in_batch == 0, pos_in_batch == max_batch_size_ - 1, true);
    }
    ReadMissingSamples();
  }

  // Get a random read sample
  LoadTargetSharedPtr ReadOne(bool is_new_batch, bool is_end_of_batch, bool dry_run = false) {
    PrepareMetadata();
    DomainTimeRange tr("[DALI][Loader] ReadOne", DomainTimeRange::kGreen1);
    // perform an initial buffer fill if it hasn't already happened
    if (!initial_buffer_filled_) {
      DomainTimeRange tr("[DALI][Loader] Filling initial buffer", DomainTimeRange::kBlue1);
      shards_.push_back({0, 0});

      // Read an initial number of samples to fill our
      // sample buffer
      for (int i = 0; i < initial_buffer_fill_; ++i) {
        LoadTargetSharedPtr tensor_ptr = nullptr;
        if (!dry_run) {
          tensor_ptr = LoadTargetSharedPtr(
            new LoadTarget,
            [this](LoadTarget* sample){
              LoadTargetUniquePtr recycle_ptr(sample);
              RecycleTensor(std::move(recycle_ptr));
            });
          PrepareEmpty(*tensor_ptr);
          ReadSample(*tensor_ptr);
        }
        sample_buffer_.push_back({read_sample_counter_, std::move(tensor_ptr)});
        IncreaseReadSampleCounter();
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
      if (ShouldPadBatch(is_new_batch)) {
        ++returned_sample_counter_;
        if (IsCheckpointingEnabled() && !ShouldPadBatch(is_end_of_batch)) {
          // If the checkpointing is enabled, the epoch is depleted and the next
          // batch will contain samples from the next epoch - increment the epoch number.
          consumer_epoch_++;
        }
        return last_sample_ptr_tmp.ptr;
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

    std::swap(sample_buffer_[idx], sample_buffer_[shards_.front().start % sample_buffer_.size()]);
    LoadTargetSharedPtr tensor_ptr = nullptr;
    if (!dry_run) {
      // now grab an empty tensor, fill it and add to filled buffers
      // empty_tensors_ needs to be thread-safe w.r.t. RecycleTensor()
      // being called by multiple consumer threads
      {
        std::lock_guard<std::mutex> lock(empty_tensors_mutex_);
        DALI_ENFORCE(empty_tensors_.size() > 0,
                     "No empty tensors - did you forget to return them?");
        tensor_ptr = {
          empty_tensors_.back().release(),
          [this](LoadTarget* sample){
            LoadTargetUniquePtr recycle_ptr(sample);
            RecycleTensor(std::move(recycle_ptr));
          }
        };
        empty_tensors_.pop_back();
      }
      ReadSample(*tensor_ptr);
    }
    IndexedLoadTargetSharedPtr sample = {read_sample_counter_, tensor_ptr};
    IncreaseReadSampleCounter();
    std::swap(sample_buffer_[shards_.back().end % sample_buffer_.size()], sample);
    ++shards_.back().end;
    last_sample_ptr_tmp = sample;

    shards_.front().start++;
    returned_sample_counter_++;

    if (IsCheckpointingEnabled() && IsEpochDepleted() && !ShouldPadBatch(is_end_of_batch)) {
      // If the checkpointing is enabled, the epoch is depleted and the next
      // batch will contain samples from the next epoch - increment the epoch number.
      consumer_epoch_++;
    }

    return sample.ptr;
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

  /**
   * @brief Advances loader position in the data source by skipping n samples.
   * @warning This generic implementation is very inefficient and should be overriden.
  */
  virtual void Skip(uint64_t n) {
    LoadTargetUniquePtr tensor_ptr;
    {
      std::lock_guard<std::mutex> lock(empty_tensors_mutex_);
      DALI_ENFORCE(empty_tensors_.size() > 0, "No empty tensors");
      tensor_ptr = std::move(empty_tensors_.back());
      empty_tensors_.pop_back();
    }
    for (uint64_t i = 0; i < n; i++) {
      ReadSample(*tensor_ptr);
    }
    RecycleTensor(std::move(tensor_ptr));
  }

  /**
   * @brief Resets the loader to the first sample.
   * Like `Reset`, but shouldn't make any extra side-effects, i.e. calling `Rewind` 
   * multiple times should be have the same effect as calling it once.
  */
  virtual void Rewind(bool wrap_to_shard) {
    DALI_FAIL("Loader doesn't support rewinding, restoring from checkpoint is impossible");
  }

  void PrepareMetadata() {
    if (!loading_flag_) {
      std::lock_guard<std::mutex> l(prepare_metadata_mutex_);
      if (!loading_flag_) {
        PrepareMetadataImpl();
        std::atomic_thread_fence(std::memory_order_release);
        loading_flag_ = true;
        DALI_ENFORCE(num_shards_ <= Size(), make_string("The number of input samples: ", Size(),
                                        ", needs to be at least equal to the requested number of"
                                        " shards: ", num_shards_, "."));
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

  // Method for restoring state from the checkpoint in subclasses
  virtual void RestoreStateImpl(const LoaderStateSnapshot &state) {}

  // Method for saving the state to the checkpoint in subclasses
  virtual void SaveStateImpl(LoaderStateSnapshot &state) {}

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


  void ReadMissingSamples() {
    if (!initial_buffer_filled_) return;

    std::vector<IndexedLoadTargetSharedPtr*> to_read;
    if (!last_sample_ptr_tmp.ptr) {
      to_read.push_back(&last_sample_ptr_tmp);
    }
    for (auto &sample : sample_buffer_) {
      if (!sample.ptr) {
        to_read.push_back(&sample);
      }
    }

    // We can't move backwards, so samples have to be read in order
    std::sort(to_read.begin(), to_read.end(), [](auto a, auto b){ return a->idx < b->idx; });

    Rewind(stick_to_shard_);

    Index at = 0;
    LoadTargetSharedPtr last = nullptr;
    for (auto target : to_read) {
      if (target->idx < at) {
        target->ptr = last;
        continue;
      }

      Skip(target->idx - at);
      at = target->idx;

      LoadTargetSharedPtr tensor_ptr = {
        new LoadTarget,
        [this](LoadTarget* sample){
          LoadTargetUniquePtr recycle_ptr(sample);
          RecycleTensor(std::move(recycle_ptr));
        }
      };
      PrepareEmpty(*tensor_ptr);
      ReadSample(*tensor_ptr);
      last = tensor_ptr;
      target->ptr = std::move(tensor_ptr);
      at++;
    }

    Rewind(stick_to_shard_);
    Skip(read_sample_counter_);
  }

  std::vector<IndexedLoadTargetSharedPtr> sample_buffer_;

  std::vector<LoadTargetUniquePtr> empty_tensors_;

  // number of samples to initialize buffer with
  // ~1 minibatch seems reasonable
  bool shuffle_;
  const int initial_buffer_fill_;
  const int initial_empty_size_;
  const int tensor_init_bytes_;

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

  // Counts how many samples the reader have read returned in the current epoch (including padding)
  Index returned_sample_counter_ = 0;
  // If true, the last batch will be padded with the last sample so that the number
  // of samples matches batch size
  bool pad_last_batch_;
  // If true data will be always read using read function and copied instead of shared with the
  // target tensor, if false loader will try to mmap files if possible and wrap the content into
  // tensor without copy
  bool dont_use_mmap_;
  // Flag that indicates if the checkpointing is enabled. Some behaviour may depend on it,
  // for example the derived FileLabel changes the way global shuffling works for easier
  // restoration of the state.
  bool checkpointing_;
  // The epoch number the next returned sample belongs to,
  // tracked only if checkpointing is enabled
  int consumer_epoch_ = 0;
  // Batch size
  int max_batch_size_;
  // Number of data shards that were actually read by the reader
  // TODO(skarpinski) Make it private to prevent ReadSample from depending on it
  int virtual_shard_id_;
  // Keeps pointer to the last returned sample just in case it needs to be cloned
  IndexedLoadTargetSharedPtr last_sample_ptr_tmp;

  struct ShardBoundaries {
    Index start;
    Index end;
  };

  std::deque<ShardBoundaries> shards_;

 private:
  bool initial_buffer_filled_ = false;
  // Counts how many samples the reader have read already from this epoch
  Index read_sample_counter_ = 0;
};

template<typename T, typename... Args>
std::unique_ptr<T> InitLoader(const OpSpec& spec, Args&&... args) {
  std::unique_ptr<T> loader (new T(spec, std::forward<Args>(args)...));
  loader->Init();
  return loader;
}

};  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_LOADER_H_
