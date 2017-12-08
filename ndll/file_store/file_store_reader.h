#ifndef NDLL_FILE_STORE_FILE_STORE_READER_H_
#define NDLL_FILE_STORE_FILE_STORE_READER_H_

#include <map>
#include <random>
#include <string>
#include <vector>

#include "ndll/common.h"
#include "ndll/error_handling.h"

namespace ndll {

struct Sample {
    size_t nbytes;
    void *data;
};

using Options = std::map<std::string, std::string>;

class FileStoreReader {
 public:
  FileStoreReader(std::string uri, Options& options) {
    // initialize a random distribution -- this will be
    // used to pick from our sample buffer
    dis = std::uniform_int_distribution<>(0, 1048576);
  }
  virtual ~FileStoreReader() {};

  // Get a random read sample
  Sample ReadOne() {
    // perform an iniital buffer fill if it hasn't already happened
    if (!initial_buffer_filled_) {
      // Read an initial number of samples to fill our
      // sample buffer
      for (int i = 0; i < initial_buffer_fill_; ++i) {
        sample_buffer_.push_back(ReadSample());
      }

      initial_buffer_filled_ = true;
    }
    // choose the random index
    int idx = dis(e_) % sample_buffer_.size();
    Sample elem = sample_buffer_[idx];

    // swap end and idx
    std::swap(sample_buffer_[idx], sample_buffer_[sample_buffer_.size()-1]);
    // remove last element
    sample_buffer_.pop_back();

    // now add a new element from the FileStore
    sample_buffer_.push_back(ReadSample());

    return elem;
  }

  // Read an actual sample from the FileStore,
  // used to populate the sample buffer for "shuffled"
  // reads.
  virtual Sample ReadSample() = 0;

 protected:
  std::vector<Sample> sample_buffer_;

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
