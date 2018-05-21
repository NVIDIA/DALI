// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_READER_LOADER_RECORDIO_LOADER_H_
#define NDLL_PIPELINE_OPERATORS_READER_LOADER_RECORDIO_LOADER_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "ndll/pipeline/operators/reader/loader/indexed_file_loader.h"
#include "ndll/common.h"
#include "ndll/error_handling.h"

namespace ndll {

class RecordIOLoader : public IndexedFileLoader {
 public:
  explicit RecordIOLoader(const OpSpec& options)
    : IndexedFileLoader(options, false) {
    Init(options);
  }
  ~RecordIOLoader() {}

  void ReadIndexFile(const std::vector<std::string>& index_uris) override {
    std::vector<size_t> file_offsets;
    file_offsets.push_back(0);
    for (std::string& path : uris_) {
      FileStream * tmp = FileStream::Open(path);
      file_offsets.push_back(tmp->Size() + file_offsets.back());
      tmp->Close();
    }
    NDLL_ENFORCE(index_uris.size() == 1,
        "RecordIOReader supports only a single index file");
    const std::string& path = index_uris[0];
    std::ifstream index_file(path);
    std::vector<size_t> temp;
    size_t index, offset;
    while (index_file >> index >> offset) {
      temp.push_back(offset);
    }
    std::sort(temp.begin(), temp.end());
    size_t file_offset_index = 0;
    for (size_t i = 0; i < temp.size() - 1; ++i) {
      if (temp[i] >= file_offsets[file_offset_index + 1]) {
        ++file_offset_index;
      }
      indices_.push_back(std::make_tuple(temp[i] - file_offsets[file_offset_index],
                                         temp[i + 1] - temp[i],
                                         file_offset_index));
    }
    indices_.push_back(std::make_tuple(temp.back() - file_offsets[file_offset_index],
                                       file_offsets.back() - temp.back(),
                                       file_offset_index));
    index_file.close();
  }

  void ReadSample(Tensor<CPUBackend>* tensor) override {
    if (current_index_ == static_cast<size_t>(Size())) {
      current_index_ = 0;
      current_file_index_ = 0;
      current_file_.reset(FileStream::Open(uris_[current_file_index_]));
    }

    int64 seek_pos, size;
    size_t file_index;
    std::tie(seek_pos, size, file_index) = indices_[current_index_];

    tensor->Resize({size});

    int64 n_read = 0;
    while (n_read < size) {
      n_read += current_file_->Read(tensor->mutable_data<uint8_t>() + n_read,
                     size - n_read);
      if (n_read < size) {
        NDLL_ENFORCE(current_file_index_ + 1 < uris_.size(),
            "Incomplete or corrupted record files");
        current_file_.reset(FileStream::Open(uris_[++current_file_index_]));
      }
    }
    ++current_index_;
  }
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_READER_LOADER_RECORDIO_LOADER_H_
