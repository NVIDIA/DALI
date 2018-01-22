// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_READER_LOADER_INDEXED_FILE_LOADER_H_
#define NDLL_PIPELINE_OPERATORS_READER_LOADER_INDEXED_FILE_LOADER_H_

#include <vector>
#include <string>
#include <tuple>
#include <fstream>

#include "ndll/common.h"
#include "ndll/pipeline/operators/reader/loader/loader.h"
#include "ndll/util/file.h"

namespace ndll {

class IndexedFileLoader : public Loader<CPUBackend> {
 public:
  explicit IndexedFileLoader(const OpSpec& options)
    : Loader(options),
      current_file_(nullptr) {
      uris_ =
        options.GetRepeatedArgument<std::string>("path");
      NDLL_ENFORCE(!uris_.empty(),
          "No files specified.");
      std::vector<std::string> index_uris =
        options.GetRepeatedArgument<std::string>("index_path");
      ReadIndexFile(index_uris);
      size_t num_indices = indices_.size();
      current_index_ = num_indices/num_shards_ * shard_id_;
      int64 seek_pos, size;
      std::tie(seek_pos, size, current_file_index_) = indices_[current_index_];
      current_file_ = FileStream::Open(uris_[current_file_index_]);
      current_file_->Seek(seek_pos);
    }

  void ReadSample(Tensor<CPUBackend>* tensor) override {
    if (current_index_ >= indices_.size()) {
      Reset();
    }
    int64 seek_pos, size;
    size_t file_index;
    std::tie(seek_pos, size, file_index) = indices_[current_index_];
    if (file_index != current_file_index_) {
      current_file_->Close();
      current_file_ = FileStream::Open(uris_[file_index]);
      current_file_index_ = file_index;
    }
    tensor->Resize({size});
    tensor->mutable_data<uint8_t>();

    current_file_->Read(reinterpret_cast<uint8_t*>(tensor->raw_mutable_data()),
                        size);
    ++current_index_;
    return;
  }

  Index Size() override {
    return indices_.size();
  }

  virtual ~IndexedFileLoader() {
    if (current_file_ != nullptr) {
      current_file_->Close();
    }
  }

  virtual void ReadIndexFile(const std::vector<std::string>& index_uris) {
    NDLL_ENFORCE(index_uris.size() == uris_.size(),
        "Number of index files needs to match the number of data files");
    for (size_t i = 0; i < index_uris.size(); ++i) {
      std::ifstream fin(index_uris[i]);
      int64 pos, size;
      while (fin >> pos >> size) {
        indices_.push_back(std::make_tuple(pos, size, i));
      }
      fin.close();
    }
  }

 protected:
  void Reset() {
    current_index_ = 0;
    int64 seek_pos, size;
    size_t file_index;
    std::tie(seek_pos, size, file_index) = indices_[current_index_];
    if (file_index != current_file_index_) {
      current_file_->Close();
      current_file_ = FileStream::Open(uris_[file_index]);
      current_file_index_ = file_index;
    }
    current_file_->Seek(seek_pos);
  }

  std::vector<std::string> uris_;
  std::vector<std::tuple<int64, int64, size_t>> indices_;
  size_t current_index_;
  size_t current_file_index_;
  FileStream * current_file_;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_READER_LOADER_INDEXED_FILE_LOADER_H_
