// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_READER_LOADER_LOADER_H_
#define NDLL_PIPELINE_OPERATORS_READER_LOADER_LOADER_H_

#include <list>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/op_spec.h"
#include "ndll/pipeline/data/tensor.h"

namespace ndll {

template <class Backend>
class Loader {
 public:
  explicit Loader(const OpSpec& options)
    : initial_buffer_fill_(options.GetArgument<int>("initial_fill", 1024)),
      tensor_init_bytes_(options.GetArgument<int>("tensor_init_bytes", 1048576)),
      shard_id_(options.GetArgument<int>("shard_id", 0)),
      num_shards_(options.GetArgument<int>("num_shards", 1)) {
    // initialize a random distribution -- this will be
    // used to pick from our sample buffer
    dis = std::uniform_int_distribution<>(0, 1048576);
  }
  virtual ~Loader() {}

  // Get a random read sample
  Tensor<Backend>* ReadOne() {
    // perform an iniital buffer fill if it hasn't already happened
    if (!initial_buffer_filled_) {
      // Read an initial number of samples to fill our
      // sample buffer
      for (int i = 0; i < initial_buffer_fill_; ++i) {
        Tensor<Backend>* tensor = new Tensor<CPUBackend>();
        tensor->set_pinned(false);
        // Initialize tensors to a set size to limit expensive reallocations
        // using cudaMallocHost (and paired cudaFreeHost calls)
        tensor->Resize({tensor_init_bytes_});
        tensor->template mutable_data<uint8_t>();

        ReadSample(tensor);
        sample_buffer_.push_back(tensor);
      }

      // need some entries in the empty_tensors_ list
      for (int i = 0; i < initial_empty_size_; ++i) {
        Tensor<Backend>* tensor = new Tensor<CPUBackend>();
        // Force allocation for empties
        tensor->Resize({tensor_init_bytes_});
        tensor->template mutable_data<uint8_t>();

        empty_tensors_.push_back(tensor);
      }

      initial_buffer_filled_ = true;
    }
    // choose the random index
    int idx = dis(e_) % sample_buffer_.size();
    Tensor<Backend>* elem = sample_buffer_[idx];

    // swap end and idx, return the tensor to empties
    std::swap(sample_buffer_[idx], sample_buffer_[sample_buffer_.size()-1]);
    // remove last element
    sample_buffer_.pop_back();

    // now grab an empty tensor, fill it and add to filled buffers
    // empty_tensors_ needs to be thread-safe w.r.t. ReturnTensor()
    // being called by multiple consumer threads
    Tensor<Backend>* t;
    {
      std::lock_guard<std::mutex> lock(return_mutex_);
      NDLL_ENFORCE(empty_tensors_.size() > 0, "No empty tensors - did you forget to return them?");
      t = empty_tensors_.back();
      empty_tensors_.pop_back();
    }
    ReadSample(t);
    sample_buffer_.push_back(t);

    return elem;
  }

  // return a tensor to the empty pile
  // called by multiple consumer threads
  void ReturnTensor(Tensor<Backend>* tensor) {
    std::lock_guard<std::mutex> lock(return_mutex_);
    empty_tensors_.push_back(tensor);
  }

  // Read an actual sample from the FileStore,
  // used to populate the sample buffer for "shuffled"
  // reads.
  virtual void ReadSample(Tensor<Backend>* tensor) = 0;

  // Give the size of the data accessed through the Loader
  virtual Index Size() = 0;

 protected:
  std::vector<Tensor<Backend>*> sample_buffer_;

  std::list<Tensor<Backend>*> empty_tensors_;

  // number of samples to initialize buffer with
  // ~1 minibatch seems reasonable
  const int initial_buffer_fill_ = 1024;
  const int initial_empty_size_ = 1024;
  const int tensor_init_bytes_;
  bool initial_buffer_filled_ = false;

  // rng
  std::default_random_engine e_;
  std::uniform_int_distribution<> dis;

  // control return of tensors
  std::mutex return_mutex_;

  // sharding
  const int shard_id_;
  const int num_shards_;
};

};  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_READER_LOADER_LOADER_H_
