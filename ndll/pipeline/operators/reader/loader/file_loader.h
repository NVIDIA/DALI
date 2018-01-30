// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_READER_LOADER_FILE_LOADER_H_
#define NDLL_PIPELINE_OPERATORS_READER_LOADER_FILE_LOADER_H_

#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ndll/common.h"
#include "ndll/pipeline/operators/reader/loader/loader.h"
#include "ndll/util/file.h"

namespace ndll {

class FileLoader : public Loader<CPUBackend> {
 public:
  explicit FileLoader(const OpSpec& spec)
    : Loader<CPUBackend>(spec),
      file_root_(spec.GetArgument<string>("file_root", "")),
      file_list_(spec.GetArgument<string>("file_list", "")),
      current_index_(0) {
    // load (path, label) pairs from list
    std::ifstream s(file_list_);
    NDLL_ENFORCE(s.is_open());

    string image_file;
    int label;
    while (s >> image_file >> label) {
      auto p = std::make_pair(file_root_ + "/" + image_file, label);
      image_label_pairs_.push_back(p);
    }

    // first / only shard: no change needed
    if (shard_id_ == 0) return;

    int samples_per_shard = Size() / num_shards_;
    current_index_ = shard_id_ * samples_per_shard;
  }

  void ReadSample(Tensor<CPUBackend>* tensor) override {
    auto image_pair = image_label_pairs_[current_index_++];

    // handle wrap-around
    if (current_index_ == Size()) {
      current_index_ = 0;
    }

    FileStream *current_image = FileStream::Open(image_pair.first);
    Index image_size = current_image->Size();

    // resize tensor to hold [image, label]
    tensor->Resize({image_size + static_cast<Index>(sizeof(int))});

    // copy the image
    current_image->Read(tensor->mutable_data<uint8_t>(), image_size);

    // close the file handle
    current_image->Close();

    // copy the label
    tensor->mutable_data<uint8_t>()[image_size] = image_pair.second;
  }

  Index Size() override {
    return static_cast<Index>(image_label_pairs_.size());
  }

 protected:
  using Loader<CPUBackend>::shard_id_;
  using Loader<CPUBackend>::num_shards_;

  string file_root_, file_list_;
  vector<std::pair<string, int>> image_label_pairs_;
  Index current_index_;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_READER_LOADER_FILE_LOADER_H_
