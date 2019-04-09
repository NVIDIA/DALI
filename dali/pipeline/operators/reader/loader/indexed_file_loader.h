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

#ifndef DALI_PIPELINE_OPERATORS_READER_LOADER_INDEXED_FILE_LOADER_H_
#define DALI_PIPELINE_OPERATORS_READER_LOADER_INDEXED_FILE_LOADER_H_

#include <vector>
#include <string>
#include <tuple>
#include <fstream>
#include <memory>

#include "dali/common.h"
#include "dali/pipeline/operators/reader/loader/loader.h"
#include "dali/util/file.h"

namespace dali {

class IndexedFileLoader : public Loader<CPUBackend, Tensor<CPUBackend>> {
 public:
  explicit IndexedFileLoader(const OpSpec& options)
    : Loader(options),
      uris_(options.GetRepeatedArgument<std::string>("path")),
      index_uris_(options.GetRepeatedArgument<std::string>("index_path")),
      current_file_(nullptr) {
    }

  void ReadSample(Tensor<CPUBackend>& tensor) override {
    MoveToNextShard(current_index_);

    int64 seek_pos, size;
    size_t file_index;
    std::tie(seek_pos, size, file_index) = indices_[current_index_];
    ++current_index_;

    std::string image_key = uris_[file_index] + " at index " + to_string(seek_pos);
    tensor.SetSourceInfo(image_key);
    tensor.SetSkipSample(false);

    if (file_index != current_file_index_) {
      current_file_->Close();
      current_file_ = FileStream::Open(uris_[file_index], read_ahead_);
      current_file_index_ = file_index;
    }

    // if image is cached, skip loading
    if (ShouldSkipImage(image_key)) {
      tensor.set_type(TypeInfo::Create<uint8_t>());
      tensor.Resize({1});
      tensor.SetSkipSample(true);
      should_seek_ = true;
      return;
    }

    if (should_seek_) {
      current_file_->Seek(seek_pos);
      should_seek_ = false;
    }

    if (!copy_read_data_) {
      auto p = current_file_->Get(size);
      DALI_ENFORCE(p != nullptr, "Error reading from a file " + uris_[current_file_index_]);
      // Wrap the raw data in the Tensor object.
      tensor.ShareData(p, size, {size});
      tensor.set_type(TypeInfo::Create<uint8_t>());
    } else {
      tensor.set_type(TypeInfo::Create<uint8_t>());
      tensor.Resize({size});

      int64 n_read = current_file_->Read(reinterpret_cast<uint8_t*>(tensor.raw_mutable_data()),
                          size);
      DALI_ENFORCE(n_read == size, "Error reading from a file " + uris_[current_file_index_]);
    }

    return;
  }

  ~IndexedFileLoader() override {
    if (current_file_ != nullptr) {
      current_file_->Close();
    }
  }

  virtual void ReadIndexFile(const std::vector<std::string>& index_uris) {
    DALI_ENFORCE(index_uris.size() == uris_.size(),
        "Number of index files needs to match the number of data files");
    for (size_t i = 0; i < index_uris.size(); ++i) {
      std::ifstream fin(index_uris[i]);
      DALI_ENFORCE(fin.good(), "Failed to open file " + index_uris[i]);
      int64 pos, size;
      while (fin >> pos >> size) {
        indices_.push_back(std::make_tuple(pos, size, i));
      }
      fin.close();
    }
  }

 protected:
  Index SizeImpl() override {
    return indices_.size();
  }

  void PrepareMetadataImpl() override {
    DALI_ENFORCE(!uris_.empty(), "No files specified.");
    ReadIndexFile(index_uris_);
    DALI_ENFORCE(!indices_.empty(), "Content of index files should not be empty");
    current_file_index_ = INVALID_INDEX;
    Reset(true);

    mmap_reserver = FileStream::FileStreamMappinReserver(uris_.size());
    copy_read_data_ = !mmap_reserver.CanShareMappedData();
  }

  void Reset(bool wrap_to_shard) override {
    int64 seek_pos, size;
    size_t file_index;
    if (wrap_to_shard) {
      current_index_ = start_index(shard_id_, num_shards_, Size());
    } else {
      current_index_ = 0;
    }
    std::tie(seek_pos, size, file_index) = indices_[current_index_];
    if (file_index != current_file_index_) {
      if (current_file_index_ != static_cast<size_t>(INVALID_INDEX)) {
        current_file_->Close();
      }
      current_file_ = FileStream::Open(uris_[file_index], read_ahead_);
      current_file_index_ = file_index;
    }
    current_file_->Seek(seek_pos);
  }

  std::vector<std::string> uris_;
  std::vector<std::string> index_uris_;
  std::vector<std::tuple<int64, int64, size_t>> indices_;
  size_t current_index_;
  size_t current_file_index_;
  std::unique_ptr<FileStream> current_file_;
  FileStream::FileStreamMappinReserver mmap_reserver;
  static constexpr int INVALID_INDEX = -1;
  bool should_seek_ = false;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_LOADER_INDEXED_FILE_LOADER_H_
