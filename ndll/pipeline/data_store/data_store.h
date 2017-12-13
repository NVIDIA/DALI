#ifndef NDLL_DATA_STORE_DATA_STORE_H_
#define NDLL_DATA_STORE_DATA_STORE_H_

#include <list>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/op_spec.h"
#include "ndll/pipeline/data/tensor.h"

namespace ndll {

template <class Backend>
class DataStore {
 public:
  DataStore(const OpSpec& options) {
    // initialize a random distribution -- this will be
    // used to pick from our sample buffer
    dis = std::uniform_int_distribution<>(0, 1048576);
  }
  virtual ~DataStore() {};

  // Get a random read sample
  Tensor<Backend>* ReadOne() {
    // perform an iniital buffer fill if it hasn't already happened
    if (!initial_buffer_filled_) {
      // Read an initial number of samples to fill our
      // sample buffer
      for (int i = 0; i < initial_buffer_fill_; ++i) {
        Tensor<Backend>* tensor = new Tensor<CPUBackend>();
        ReadSample(tensor);
        sample_buffer_.push_back(tensor);
      }

      // need some entries in the empty_tensors_ list
      for (int i = 0; i < 10; ++i) {
        Tensor<Backend>* tensor = new Tensor<CPUBackend>();
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
    NDLL_ENFORCE(empty_tensors_.size() > 0, "No empty tensors - did you forget to return them?");
    Tensor<Backend>* t = empty_tensors_.back();
    empty_tensors_.pop_back();
    ReadSample(t);
    sample_buffer_.push_back(t);

    return elem;
  }

  void ReturnTensor(Tensor<Backend>* tensor) {
    empty_tensors_.push_back(tensor);
  }

  // Read an actual sample from the FileStore,
  // used to populate the sample buffer for "shuffled"
  // reads.
  virtual void ReadSample(Tensor<Backend>* tensor) = 0;

  // Give the size of the data accessed through the DataStore
  virtual Index Size() = 0;

 protected:
  std::vector<Tensor<Backend>*> sample_buffer_;

  std::list<Tensor<Backend>*> empty_tensors_;

  // number of samples to initialize buffer with
  // ~1 minibatch seems reasonable
  const int initial_buffer_fill_ = 1024;
  bool initial_buffer_filled_ = false;

  // rng
  std::default_random_engine e_;
  std::uniform_int_distribution<> dis;
};

}; // namespace ndll

#endif
