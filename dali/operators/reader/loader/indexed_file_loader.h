// Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_LOADER_INDEXED_FILE_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_INDEXED_FILE_LOADER_H_

#include <vector>
#include <string>
#include <tuple>
#include <fstream>
#include <memory>
#include <queue>
#include <mutex>
#include <utility>

#include "dali/core/common.h"
#include "dali/core/mm/memory.h"
#include "dali/operators/reader/loader/loader.h"
#include "dali/util/file.h"
#include "dali/util/odirect_file.h"

namespace dali {

class IndexedFileLoader : public Loader<CPUBackend, Tensor<CPUBackend>, true> {
 public:
  explicit IndexedFileLoader(const OpSpec& spec)
    : Loader(spec),
      uris_(spec.GetRepeatedArgument<std::string>("path")),
      index_uris_(spec.GetRepeatedArgument<std::string>("index_path")),
      current_index_(0), current_file_index_(0), current_file_(nullptr),
      use_o_direct_(spec.HasArgument("use_o_direct") && spec.GetArgument<bool>("use_o_direct")) {
        DALI_ENFORCE(dont_use_mmap_  || !use_o_direct_, make_string("Cannot use use_o_direct with ",
                     "``dont_use_mmap=False``."));
      if (use_o_direct_) {
        o_direct_chunk_size_ = ODirectFileStream::GetChunkSize();
        o_direct_alignm_ = ODirectFileStream::GetAlignment();
        o_direct_read_len_alignm_ = ODirectFileStream::GetLenAlignment();
      }
    }

  void ReadSample(Tensor<CPUBackend>& tensor) override {
    MoveToNextShard(current_index_);

    int64 seek_pos, size;
    size_t file_index;
    std::tie(seek_pos, size, file_index) = indices_[current_index_];
    ++current_index_;

    std::string image_key = uris_[file_index] + " at index " + to_string(seek_pos);
    DALIMeta meta;
    meta.SetSourceInfo(image_key);
    meta.SetSkipSample(false);

    if (file_index != current_file_index_) {
      current_file_.reset();
      current_file_ = FileStream::Open(uris_[file_index], read_ahead_, !copy_read_data_,
                                       use_o_direct_);
      current_file_index_ = file_index;
      // invalidate the buffer
      if (use_o_direct_) read_buffer_.reset();
    }

    // if image is cached, skip loading
    if (ShouldSkipImage(image_key)) {
      meta.SetSkipSample(true);
      should_seek_ = true;
      tensor.Reset();
      tensor.SetMeta(meta);
      tensor.Resize({0}, DALI_UINT8);
      return;
    }

    if (should_seek_ || next_seek_pos_ != seek_pos) {
      current_file_->SeekRead(seek_pos);
      should_seek_ = false;
    }
    next_seek_pos_ = seek_pos + size;

    if (!copy_read_data_) {
      auto p = current_file_->Get(size);
      DALI_ENFORCE(p != nullptr, "Error reading from a file " + uris_[current_file_index_]);
      // Wrap the raw data in the Tensor object.
      tensor.ShareData(p, size, false, {size}, DALI_UINT8, CPU_ONLY_DEVICE_ID);
    } else {
      if (tensor.shares_data()) {
        tensor.Reset();
      }
      if (use_o_direct_) {
        /*
         *   ** - sample data
         *   XX - buffer padding, data of other samples
         *
         *   <--     TFRecord file                                           -->
         *   |               <-  read_buffer_                 ->               |
         *   |<-    seek_pos   -><- size   ->                  |               |
         *   |<-block_start ->   |          |                  |               |
         *   |<-             |  block_end   |                 ->               |
         *   |               <- aligned_len/read_buffer_size_ ->               |
         *   ----------------XXXX************XXXXXXXXXXXXXXXXXXX----------------
         */
        // read again if there is no buffer of the requested piece if outside of the it
        bool after_buffer_start = seek_pos >= static_cast<int64>(read_buffer_pos_);
        bool before_buffer_end = seek_pos + size <
                                    static_cast<int64>(read_buffer_pos_ + read_buffer_data_size_);
        // buffer need to exists and the ata we look for needs to be inside it
        if (!read_buffer_ || !(after_buffer_start && before_buffer_end)) {
          // check how much we need to allocate to house the required sample, but no less than
          // o_direct_chunk_size_
          auto block_start = align_down(seek_pos, o_direct_alignm_);
          auto block_end = align_up(seek_pos + size, o_direct_alignm_);
          auto aligned_len = align_up(block_end - block_start, o_direct_chunk_size_);
          // make the staging buffer as big as the biggest sample so far
          if (aligned_len > static_cast<int64>(read_buffer_size_)) {
            read_buffer_size_ = aligned_len;
          }
          // the old memory will be used as long as any piece of it uses its, so it is safe
          // to release the old buffer from read_buffer_
          read_buffer_ = mm::alloc_raw_shared<char, mm::memory_kind::host>(read_buffer_size_,
                                                                           o_direct_alignm_);
          read_buffer_pos_ = block_start;
          read_buffer_data_size_ = aligned_len;
          auto file_name = uris_[file_index];
          auto file = dynamic_cast<ODirectFileStream*>(current_file_.get());
          auto o_direct_chunk_size_tmp = o_direct_chunk_size_;
          // capture shared ptr to file in lambda to make sure it is alive as long as we want to
          // access it in any piece of work and it is not closed
          shared_ptr<FileStream> tmp_file_ptr = current_file_;
          // split reads into chunks
          for (size_t read_off = 0; static_cast<size_t>(aligned_len) > read_off;
               read_off += o_direct_chunk_size_) {
            auto dst_ptr = read_buffer_.get() + read_off;
            auto read_start = block_start + read_off;
            // we should read either the chunk size or the reminder of the file
            auto min_read = std::min(o_direct_chunk_size_tmp, seek_pos + size - read_start);
            auto work = [tmp_file_ptr, file, dst_ptr, o_direct_chunk_size_tmp, min_read,
                         read_start, file_name]() {
              auto ret = file->ReadAt(dst_ptr, o_direct_chunk_size_tmp, read_start);
              DALI_ENFORCE(ret >= min_read && ret <= o_direct_chunk_size_tmp,
                           make_string("Failed to read file: ", file_name,
                                       ", read: ", ret, " while it should be in range [", min_read,
                                       ", ", o_direct_chunk_size_tmp, "]"));
            };
            // store the work lambda into queue so the prefetch thread can pick them up latter and
            // execute in multiple threads
            PutReadWork(std::move(work));
          }
        }
        shared_ptr<void> tmp_mem(read_buffer_, read_buffer_.get() + (seek_pos - read_buffer_pos_));
        // make sure it is a big value in signed range
        tensor.ShareData(tmp_mem, size, false, {size}, DALI_UINT8, -1);
      } else {
        tensor.Resize({size}, DALI_UINT8);

        int64 n_read = current_file_->Read(reinterpret_cast<uint8_t*>(tensor.raw_mutable_data()),
                            size);
        DALI_ENFORCE(n_read == size, "Error reading from a file " + uris_[current_file_index_]);
      }
    }

    tensor.SetMeta(meta);
    return;
  }

  void Skip() override {
    MoveToNextShard(current_index_++);
  }

  ~IndexedFileLoader() override {
    current_file_.reset();
  }

  virtual void ReadIndexFile(const std::vector<std::string>& index_uris) {
    DALI_ENFORCE(index_uris.size() == uris_.size(),
        "Number of index files needs to match the number of data files");
    for (size_t i = 0; i < index_uris.size(); ++i) {
      std::ifstream fin(index_uris[i]);
      DALI_ENFORCE(fin.good(), "Failed to open file " + index_uris[i]);
      int64 pos, size;
      while (fin >> pos >> size) {
        indices_.emplace_back(pos, size, i);
      }
      fin.close();
    }
  }

 protected:
  Index SizeImpl() override {
    return indices_.size();
  }

  void PrepareMetadataImpl() override {
    if (!dont_use_mmap_) {
      mmap_reserver_ = FileStream::MappingReserver(
                                  static_cast<unsigned int>(initial_buffer_fill_));
    }
    copy_read_data_ = dont_use_mmap_ || !mmap_reserver_.CanShareMappedData();

    DALI_ENFORCE(!uris_.empty(), "No files specified.");
    ReadIndexFile(index_uris_);
    DALI_ENFORCE(!indices_.empty(), "Content of index files should not be empty");
    current_file_index_ = INVALID_INDEX;
    Reset(true);
  }

  void Reset(bool wrap_to_shard) override {
    int64 seek_pos, size;
    size_t file_index;
    if (wrap_to_shard) {
      current_index_ = start_index(virtual_shard_id_, num_shards_, SizeImpl());
    } else {
      current_index_ = 0;
    }
    std::tie(seek_pos, size, file_index) = indices_[current_index_];
    if (file_index != current_file_index_) {
      current_file_.reset();
      current_file_ = FileStream::Open(uris_[file_index], read_ahead_, !copy_read_data_,
                                       use_o_direct_);
      current_file_index_ = file_index;
      // invalidate the buffer
      if (use_o_direct_) read_buffer_.reset();
    }
    current_file_->SeekRead(seek_pos);
  }

  std::vector<std::string> uris_;
  std::vector<std::string> index_uris_;
  std::vector<std::tuple<int64, int64, size_t>> indices_;
  size_t current_index_;
  size_t current_file_index_;
  std::shared_ptr<FileStream> current_file_;
  FileStream::MappingReserver mmap_reserver_;
  static constexpr int INVALID_INDEX = -1;
  bool should_seek_ = false;
  int64 next_seek_pos_ = 0;
  bool use_o_direct_ = false;
  size_t o_direct_chunk_size_ = 0;
  size_t o_direct_alignm_ = 0;
  size_t o_direct_read_len_alignm_ = 0;
  shared_ptr<char> read_buffer_;
  size_t read_buffer_pos_ = 0;
  size_t read_buffer_size_ = 0;
  size_t read_buffer_data_size_ = 0;

  typedef std::function<void(void)> ReadWork;
  std::queue<ReadWork> jobs_;
  std::mutex mutex_;

  void PutReadWork(ReadWork work) {
    std::lock_guard<std::mutex> lock(mutex_);
    jobs_.push(std::move(work));
  }

 public:
  ReadWork GetReadWork() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto work = std::move(jobs_.front());
    jobs_.pop();
    return work;
  }

  bool AnyWorkLeft() {
    std::lock_guard<std::mutex> lock(mutex_);
    return jobs_.size();
  }
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_INDEXED_FILE_LOADER_H_
