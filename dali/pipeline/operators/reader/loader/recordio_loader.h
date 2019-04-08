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

#ifndef DALI_PIPELINE_OPERATORS_READER_LOADER_RECORDIO_LOADER_H_
#define DALI_PIPELINE_OPERATORS_READER_LOADER_RECORDIO_LOADER_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "dali/pipeline/operators/reader/loader/indexed_file_loader.h"
#include "dali/common.h"
#include "dali/error_handling.h"

namespace dali {

class RecordIOLoader : public IndexedFileLoader {
 public:
  explicit RecordIOLoader(const OpSpec& options)
    : IndexedFileLoader(options) {
  }
  ~RecordIOLoader() override {}

  void ReadIndexFile(const std::vector<std::string>& index_uris) override {
    std::vector<size_t> file_offsets;
    file_offsets.push_back(0);
    for (std::string& path : uris_) {
      auto tmp = FileStream::Open(path, read_ahead_);
      file_offsets.push_back(tmp->Size() + file_offsets.back());
      tmp->Close();
    }
    DALI_ENFORCE(index_uris.size() == 1,
        "RecordIOReader supports only a single index file");
    const std::string& path = index_uris[0];
    std::ifstream index_file(path);
    DALI_ENFORCE(index_file.good(),
        "Could not open RecordIO index file. Provided path: \"" + path + "\"");
    std::vector<size_t> temp;
    size_t index, offset, prev_offset = -1;
    while (index_file >> index >> offset) {
      temp.push_back(offset);
    }
    std::sort(temp.begin(), temp.end());
    size_t file_offset_index = 0;
    for (size_t i = 0; i < temp.size() - 1; ++i) {
      if (temp[i] >= file_offsets[file_offset_index + 1]) {
        ++file_offset_index;
      }
      int64 size = temp[i + 1] - temp[i];
      // skip 0 sized images
      if (size) {
        indices_.push_back(std::make_tuple(temp[i] - file_offsets[file_offset_index],
                                          size, file_offset_index));
      }
    }
    int64 size = file_offsets.back() - temp.back();
    // skip 0 sized images
    if (size) {
      indices_.push_back(std::make_tuple(temp.back() - file_offsets[file_offset_index],
                                        size, file_offset_index));
    }
    index_file.close();
  }

  void ReadSample(Tensor<CPUBackend>& tensor) override {
    // if we moved to next shard wrap up
    MoveToNextShard(current_index_);

    int64 seek_pos, size;
    size_t file_index;
    std::tie(seek_pos, size, file_index) = indices_[current_index_];

    ++current_index_;

    std::string image_key = uris_[file_index] + " at index " + to_string(seek_pos);
    tensor.SetSourceInfo(image_key);
    tensor.SetSkipSample(false);

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

    shared_ptr<void> p = nullptr;
    int64 n_read = 0;
    bool use_read = copy_read_data_;
    if (use_read) {
      tensor.Resize({size});
    }
    while (p == nullptr && n_read < size) {
      if (!use_read) {
        p = current_file_->Get(size);
        // file is divided between two files, we need to fallback to read here
        if (p == nullptr) {
          tensor.Resize({size});
          use_read = true;
        } else {
          n_read = size;
          // Wrap the raw data in the Tensor object.
          tensor.ShareData(p, size, {size});
          tensor.set_type(TypeInfo::Create<uint8_t>());
        }
      }
      if (use_read) {
        n_read += current_file_->Read(tensor.mutable_data<uint8_t>() + n_read,
                                      size - n_read);
      }
      if (p == nullptr && n_read < size) {
        DALI_ENFORCE(current_file_index_ + 1 < uris_.size(),
          "Incomplete or corrupted record files");
        // Release previously opened file
        current_file_ = FileStream::Open(uris_[++current_file_index_], read_ahead_);
        continue;
      }
    }
  }

 private:
  bool should_seek_ = false;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_LOADER_RECORDIO_LOADER_H_
