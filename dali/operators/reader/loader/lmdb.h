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

#ifndef DALI_OPERATORS_READER_LOADER_LMDB_H_
#define DALI_OPERATORS_READER_LOADER_LMDB_H_

#include <lmdb.h>
#include <memory>
#include <string>
#include <vector>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/loader.h"

namespace dali {

#define CHECK_LMDB(status, filename) \
  do { \
    DALI_ENFORCE(status == MDB_SUCCESS, "LMDB Error: " + string(mdb_strerror(status)) + \
                                        ", with file: " + filename); \
  } while (0)


class IndexedLMDB {
  MDB_env* mdb_env_ = nullptr;
  MDB_cursor* mdb_cursor_ = nullptr;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_transaction_ = nullptr;
  int num_;
  Index mdb_index_;
  std::string db_path_;
  Index mdb_size_;

 public:
  void Open(const std::string& path, int num) {
    DALI_ENFORCE(mdb_env_ == nullptr, "Previous MDB environment was not closed");
    db_path_ = path;
    num_ = num;
    CHECK_LMDB(mdb_env_create(&mdb_env_), db_path_);
    auto mdb_flags = MDB_RDONLY | MDB_NOTLS | MDB_NOLOCK;
    CHECK_LMDB(mdb_env_open(mdb_env_, path.c_str(), mdb_flags, 0664), db_path_);

    // Create transaction and cursor
    CHECK_LMDB(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_transaction_), db_path_);
    CHECK_LMDB(mdb_dbi_open(mdb_transaction_, NULL, 0, &mdb_dbi_), db_path_);
    CHECK_LMDB(mdb_cursor_open(mdb_transaction_, mdb_dbi_, &mdb_cursor_), db_path_);
    MDB_stat stat;
    CHECK_LMDB(mdb_stat(mdb_transaction_, mdb_dbi_, &stat), db_path_);
    mdb_size_ = stat.ms_entries;
    LOG_LINE << "lmdb " << num_ << " " << db_path_
             << " has " << mdb_size_ << " entries" << std::endl;
    mdb_index_ = 0;
  }
  size_t GetSize() const { return mdb_size_; }
  Index GetIndex() const { return mdb_index_; }
  void SeekByIndex(Index index, MDB_val* key = nullptr, MDB_val* value = nullptr) {
    MDB_val tmp_key, tmp_value;
    if (nullptr == key) {
      key = &tmp_key;
    }
    if (nullptr == value) {
      value = &tmp_value;
    }
    if (index == 0) {
      LOG_LINE << "lmdb " << num_ << " " << db_path_
               << " rewind to the begin from " << mdb_index_ << std::endl;
    }
    DALI_ENFORCE(index >= 0 && index < mdb_size_);
    if (index == 0) {
      CHECK_LMDB(mdb_cursor_get(mdb_cursor_, key, value, MDB_FIRST), db_path_);
    } else if (index == mdb_size_ - 1) {
      CHECK_LMDB(mdb_cursor_get(mdb_cursor_, key, value, MDB_LAST), db_path_);
    } else if (index == mdb_index_) {
      CHECK_LMDB(mdb_cursor_get(mdb_cursor_, key, value, MDB_GET_CURRENT), db_path_);
    } else if (index == mdb_index_ - 1) {
      CHECK_LMDB(mdb_cursor_get(mdb_cursor_, key, value, MDB_PREV), db_path_);
    } else if (index == mdb_index_ + 1) {
      CHECK_LMDB(mdb_cursor_get(mdb_cursor_, key, value, MDB_NEXT), db_path_);
    } else if (index > mdb_index_) {
      LOG_LINE << "lmdb " << num_ << " " << db_path_
               << " exec a large step forward " << mdb_index_ << "->" << index << std::endl;
      for (Index i = mdb_index_; i < index; ++i) {
        CHECK_LMDB(mdb_cursor_get(mdb_cursor_, key, value, MDB_NEXT), db_path_);
      }
    } else {
      // index < mdb_index_
      LOG_LINE << "lmdb " << num_ << " " << db_path_
               << " exec a large step backward " << mdb_index_ << "->" << index << std::endl;
      for (Index i = index; i < mdb_index_; i++) {
        CHECK_LMDB(mdb_cursor_get(mdb_cursor_, key, value, MDB_PREV), db_path_);
      }
    }
    mdb_index_ = index;
  }

  void Close() {
    if (mdb_cursor_) {
      mdb_cursor_close(mdb_cursor_);
      mdb_dbi_close(mdb_env_, mdb_dbi_);
      mdb_cursor_ = nullptr;
    }
    if (mdb_transaction_) {
      mdb_txn_abort(mdb_transaction_);
      mdb_transaction_ = nullptr;
    }
    if (mdb_env_) {
      mdb_env_close(mdb_env_);
      mdb_env_ = nullptr;
    }
  }
};

static int find_lower_bound(const std::vector<Index>& a, Index x) {
  DALI_ENFORCE(x >= a.front() && x < a.back() && a.size() >= 2);
  int low = 0;
  int high = a.size()-2;
  do {
    int mid = (low+high) / 2;
    if (x >= a[mid] && x < a[mid+1]) {
      return mid;
    } else if (x >= a[mid+1]) {
      low = mid+1;
    } else if (x < a[mid]) {
      high = mid-1;
    }
  } while (low <= high);

  DALI_FAIL("the array is not in ascending order.");
  return -1;
}

class LMDBLoader : public Loader<CPUBackend, Tensor<CPUBackend>> {
 public:
  explicit LMDBLoader(const OpSpec& options)
      : Loader(options) {
    bool ret = options.TryGetRepeatedArgument<std::string>(db_paths_, "path");
    if (!ret) {
      std::string path = options.GetArgument<std::string>("path");
      db_paths_.push_back(path);
    }
  }

  ~LMDBLoader() override {
    for (size_t i = 0; i < mdb_.size(); i++) {
      mdb_[i].Close();
    }
  }

  void MapIndexToFile(Index index, Index& file_index, Index& local_index) {
    DALI_ENFORCE(offsets_.size() > 0);
    DALI_ENFORCE(index >= 0 && index < offsets_.back());
    file_index = find_lower_bound(offsets_, index);
    local_index = index - offsets_[file_index];
  }

  void ReadSample(Tensor<CPUBackend>& tensor) override {
    // assume cursor is valid, read next, loop to start if necessary

    Index file_index, local_index;
    MapIndexToFile(current_index_, file_index, local_index);

    MDB_val key, value;
    mdb_[file_index].SeekByIndex(local_index, &key, &value);
    ++current_index_;

    MoveToNextShard(current_index_);

    std::string image_key = db_paths_[file_index] + " at key " +
                            to_string(reinterpret_cast<char*>(key.mv_data));
    DALIMeta meta;

    meta.SetSourceInfo(image_key);
    meta.SetSkipSample(false);

    tensor.set_type(TypeInfo::Create<uint8_t>());

    // if image is cached, skip loading
    if (ShouldSkipImage(image_key)) {
      meta.SetSkipSample(true);
      tensor.Reset();
      tensor.SetMeta(meta);
      tensor.set_type(TypeInfo::Create<uint8_t>());
      tensor.Resize({0});
      return;
    }

    tensor.SetMeta(meta);
    tensor.Resize({static_cast<Index>(value.mv_size)});
    std::memcpy(tensor.raw_mutable_data(),
                reinterpret_cast<uint8_t*>(value.mv_data),
                value.mv_size * sizeof(uint8_t));
  }

 protected:
  Index SizeImpl() override {
    return offsets_.size() > 0 ? offsets_.back() : 0;
  }

  void PrepareMetadataImpl() override {
    offsets_.resize(db_paths_.size() + 1);
    offsets_[0] = 0;
    mdb_.resize(db_paths_.size());
    for (size_t i = 0; i < db_paths_.size(); i++) {
      mdb_[i].Open(db_paths_[i], i);
      offsets_[i + 1] = offsets_[i] + mdb_[i].GetSize();
    }
    Reset(true);
  }

 private:
  void Reset(bool wrap_to_shard) override {
    // work out how many entries to move forward to handle sharding
    if (wrap_to_shard) {
      current_index_ = start_index(shard_id_, num_shards_, SizeImpl());
    } else {
      current_index_ = 0;
    }
    Index file_index, local_index;
    MapIndexToFile(current_index_, file_index, local_index);

    mdb_[file_index].SeekByIndex(local_index);
  }
  using Loader<CPUBackend, Tensor<CPUBackend>>::shard_id_;
  using Loader<CPUBackend, Tensor<CPUBackend>>::num_shards_;

  std::vector<IndexedLMDB> mdb_;

  Index current_index_ = 0;

  std::vector<Index> offsets_;

  // options
  std::vector<std::string> db_paths_;
};

};  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_LMDB_H_
