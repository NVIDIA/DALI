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
  explicit IndexedFileLoader(const OpSpec& options, bool init = true)
    : Loader(options),
      current_file_(nullptr) {
      // trick for https://stackoverflow.com/questions/962132/calling-virtual-functions-inside-constructors
      if (init)
        Init(options);
    }

  void ReadSample(Tensor<CPUBackend>* tensor) override {
    if (current_index_ == indices_.size()) {
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

    int64 n_read = current_file_->Read(reinterpret_cast<uint8_t*>(tensor->raw_mutable_data()),
                        size);
    tensor->SetSourceInfo(uris_[current_file_index_] + " at index " + to_string(seek_pos));
    DALI_ENFORCE(n_read == size, "Error reading from a file");
    ++current_index_;
    return;
  }

  Index Size() override {
    return indices_.size();
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
  void Init(const OpSpec& options) {
    uris_ =
      options.GetRepeatedArgument<std::string>("path");
    DALI_ENFORCE(!uris_.empty(),
        "No files specified.");
    std::vector<std::string> index_uris =
      options.GetRepeatedArgument<std::string>("index_path");
    ReadIndexFile(index_uris);
    size_t num_indices = indices_.size();
    current_index_ = start_index(shard_id_, num_shards_, num_indices);
    int64 seek_pos, size;
    std::tie(seek_pos, size, current_file_index_) = indices_[current_index_];
    current_file_ = FileStream::Open(uris_[current_file_index_]);
    current_file_->Seek(seek_pos);
  }

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
  std::unique_ptr<FileStream> current_file_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_LOADER_INDEXED_FILE_LOADER_H_
