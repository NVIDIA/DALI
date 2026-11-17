// Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_LOADER_RECORDIO_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_RECORDIO_LOADER_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dali/operators/reader/loader/indexed_file_loader.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/util/uri.h"

namespace dali {

class RecordIOLoader : public IndexedFileLoader {
 public:
  explicit RecordIOLoader(const OpSpec& options)
    : IndexedFileLoader(options) {
  }
  ~RecordIOLoader() override {}

  void ReadIndexFile(const std::vector<std::string>& index_paths) override {
    std::vector<size_t> file_offsets;
    file_offsets.push_back(0);
    for (std::string& path : paths_) {
      FileStream::Options opts;
      opts.read_ahead = read_ahead_;
      // Only the size is needed here. Empty shards cannot be memory-mapped.
      opts.use_mmap = false;
      opts.use_odirect = false;
      auto tmp = FileStream::Open(path, opts);
      auto file_size = tmp->Size();
      // Record spans are stored in int64_t. Bound the sum even when individual sizes fit.
      if (file_size > static_cast<size_t>(std::numeric_limits<int64_t>::max()) -
                          file_offsets.back()) {
        throw std::runtime_error(make_string(
            "RecordIO data files are too large: ", file_offsets.back(), " bytes plus ",
            file_size, " bytes exceeds the range of a signed 64-bit integer."));
      }
      file_offsets.push_back(file_size + file_offsets.back());
      tmp->Close();
    }
    DALI_ENFORCE(index_paths.size() == 1,
        "RecordIOReader supports only a single index file");
    const std::string& path = index_paths[0];
    // FileStream::Open dispatches on the URI scheme (e.g. s3://), so the index file can live
    // next to the shards it describes, remote or local, same as indexed_file_loader.h.
    auto index_file = FileStream::Open(path);
    auto index_file_cleanup = AtScopeExit([&index_file] {
      if (index_file)
        index_file->Close();
    });
    FileStreamBuf<> stream_buf(index_file.get());
    std::istream fin(&stream_buf);
    DALI_ENFORCE(fin.good(),
        "Could not open RecordIO index file. Provided path: \"" + path + "\"");
    std::vector<size_t> temp;
    size_t entry_number = 0;
    while (true) {
      fin >> std::ws;
      if (fin.eof())
        break;
      size_t index = 0, offset = 0;
      ++entry_number;
      if (!(fin >> index >> offset)) {
        throw std::runtime_error(make_string(
            "Invalid RecordIO index entry ", entry_number, " in ", path,
            "; expected two integers (record ID and offset)."));
      }
      if (offset >= file_offsets.back()) {
        throw std::runtime_error(make_string(
            "RecordIO index offset out of bounds at entry ", entry_number, " in ", path,
            ": got ", offset, " bytes, expected less than ", file_offsets.back(), " bytes."));
      }
      temp.push_back(offset);
    }
    DALI_ENFORCE(!temp.empty(),
      make_string("RecordIO index file doesn't contain any indices. Provided path: \"",
                  path, "\""));

    std::sort(temp.begin(), temp.end());
    size_t file_offset_index = 0;
    for (size_t i = 0; i < temp.size(); ++i) {
      auto next_offset = i + 1 < temp.size() ? temp[i + 1] : file_offsets.back();
      int64_t size = next_offset - temp[i];
      // Skip duplicate offsets. Both offset arrays are sorted, so advance across skipped
      // or empty files at most once, including when locating the last record.
      if (size) {
        while (file_offset_index + 1 < paths_.size() &&
               temp[i] >= file_offsets[file_offset_index + 1]) {
          ++file_offset_index;
        }
        indices_.emplace_back(temp[i] - file_offsets[file_offset_index],
                              size, file_offset_index);
      }
    }
  }

  void ReadSample(IndexedFileLoaderSample& sample) override {
    // if we moved to next shard wrap up
    MoveToNextShard(current_index_);

    int64_t seek_pos, size;
    size_t file_index;
    std::tie(seek_pos, size, file_index) = indices_[current_index_];

    ++current_index_;

    std::string image_key = paths_[file_index] + " at index " + to_string(seek_pos);
    DALIMeta meta;
    meta.SetSourceInfo(image_key);
    meta.SetSkipSample(false);

    // if image is cached, skip loading
    if (ShouldSkipImage(image_key)) {
      meta.SetSkipSample(true);
      should_seek_ = true;
      sample.tensor.Reset();
      sample.tensor.SetMeta(meta);
      sample.tensor.Resize({0}, DALI_UINT8);
      return;
    }

    // Reopen the correct file when the index jumps to a different one.
    // This is needed when shuffle_after_epoch reorders records across files.
    if (file_index != current_file_index_) {
      current_file_.reset();
      FileStream::Options switch_opts;
      switch_opts.read_ahead = read_ahead_;
      switch_opts.use_mmap = !copy_read_data_;
      switch_opts.use_odirect = false;
      current_file_ = FileStream::Open(paths_[file_index], switch_opts);
      current_file_index_ = file_index;
      should_seek_ = true;
    }

    if (should_seek_ || next_seek_pos_ != seek_pos) {
      current_file_->SeekRead(seek_pos);
      should_seek_ = false;
    }

    shared_ptr<void> p = nullptr;
    int64_t n_read = 0;
    bool use_read = copy_read_data_ || !current_file_->CanMemoryMap();
    if (use_read) {
      sample.tensor.Resize({size});
    }
    while (p == nullptr && n_read < size) {
      if (!use_read) {
        p = current_file_->Get(size);
        // file is divided between two files, we need to fallback to read here
        if (p == nullptr) {
          if (sample.tensor.shares_data()) {
            sample.tensor.Reset();
          }
          sample.tensor.Resize({size}, DALI_UINT8);
          use_read = true;
        } else {
          n_read = size;
          // Wrap the raw data in the Tensor object.
          sample.tensor.ShareData(p, size, false, {size}, DALI_UINT8, CPU_ONLY_DEVICE_ID);
          next_seek_pos_ = seek_pos + size;
        }
      }
      if (use_read) {
        n_read += current_file_->Read(sample.tensor.mutable_data<uint8_t>() + n_read,
                                      size - n_read);
        next_seek_pos_ = seek_pos + n_read;
      }
      if (p == nullptr && n_read < size) {
        DALI_ENFORCE(current_file_index_ + 1 < paths_.size(),
          "Incomplete or corrupted record files");
        const auto& path = paths_[++current_file_index_];
        FileStream::Options opts;
        opts.read_ahead = read_ahead_;
        opts.use_mmap = !copy_read_data_;
        opts.use_odirect = false;
        // Release previously opened file
        current_file_ = FileStream::Open(path, opts);
        next_seek_pos_ = 0;
        continue;
      }
    }
    sample.tensor.SetMeta(meta);
  }
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_RECORDIO_LOADER_H_
