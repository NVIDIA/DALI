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

#ifndef DALI_OPERATORS_READER_LOADER_FILE_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_FILE_LOADER_H_

#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>

#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <algorithm>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/loader.h"
#include "dali/util/file.h"

namespace dali {

namespace filesystem {

vector<std::pair<string, int>> traverse_directories(const std::string& path);

}  // namespace filesystem

struct ImageLabelWrapper {
  Tensor<CPUBackend> image;
  int label;
};

class FileLoader : public Loader<CPUBackend, ImageLabelWrapper> {
 public:
  explicit inline FileLoader(
    const OpSpec& spec,
    vector<std::pair<string, int>> image_label_pairs = std::vector<std::pair<string, int>>(),
    bool shuffle_after_epoch = false)
    : Loader<CPUBackend, ImageLabelWrapper>(spec),
      file_root_(spec.GetArgument<string>("file_root")),
      file_list_(spec.GetArgument<string>("file_list")),
      image_label_pairs_(std::move(image_label_pairs)),
      shuffle_after_epoch_(shuffle_after_epoch),
      current_index_(0),
      current_epoch_(0) {
      /*
      * Those options are mutually exclusive as `shuffle_after_epoch` will make every shard looks differently
      * after each epoch so coexistence with `stick_to_shard` doesn't make any sense
      * Still when `shuffle_after_epoch` we will set `stick_to_shard` internally in the FileLoader so all
      * DALI instances will do shuffling after each epoch
      */
      if (shuffle_after_epoch_ || stick_to_shard_)
        DALI_ENFORCE(
          !shuffle_after_epoch_ || !stick_to_shard_,
          "shuffle_after_epoch and stick_to_shard cannot be both true");
      if (shuffle_after_epoch_ || shuffle_)
        DALI_ENFORCE(
          !shuffle_after_epoch_ || !shuffle_,
          "shuffle_after_epoch and random_shuffle cannot be both true");
      /*
       * Imply `stick_to_shard` from  `shuffle_after_epoch`
       */
      if (shuffle_after_epoch_) {
        stick_to_shard_ = true;
      }
    mmap_reserver = FileStream::FileStreamMappinReserver(
        static_cast<unsigned int>(initial_buffer_fill_));
    copy_read_data_ = !mmap_reserver.CanShareMappedData();
  }

  void PrepareEmpty(ImageLabelWrapper &tensor) override;
  void ReadSample(ImageLabelWrapper &tensor) override;

 protected:
  Index SizeImpl() override;

  void PrepareMetadataImpl() override {
    if (image_label_pairs_.empty()) {
      if (file_list_ == "") {
        image_label_pairs_ = filesystem::traverse_directories(file_root_);
      } else {
        // load (path, label) pairs from list
        std::ifstream s(file_list_);
        DALI_ENFORCE(s.is_open(), "Cannot open: " + file_list_);

        string image_file;
        int label;
        while (s >> image_file >> label) {
          auto p = std::make_pair(image_file, label);
          image_label_pairs_.push_back(p);
        }
        DALI_ENFORCE(s.eof(), "Wrong format of file_list: " + file_list_);
      }
    }
    DALI_ENFORCE(Size() > 0, "No files found.");

    if (shuffle_) {
      // seeded with hardcoded value to get
      // the same sequence on every shard
      std::mt19937 g(kDaliDataloaderSeed);
      std::shuffle(image_label_pairs_.begin(), image_label_pairs_.end(), g);
    }
    Reset(true);
  }

  void Reset(bool wrap_to_shard) override {
    if (wrap_to_shard) {
      current_index_ = start_index(shard_id_, num_shards_, Size());
    } else {
      current_index_ = 0;
    }

    current_epoch_++;

    if (shuffle_after_epoch_) {
      std::mt19937 g(kDaliDataloaderSeed + current_epoch_);
      std::shuffle(image_label_pairs_.begin(), image_label_pairs_.end(), g);
    }
  }

  using Loader<CPUBackend, ImageLabelWrapper>::shard_id_;
  using Loader<CPUBackend, ImageLabelWrapper>::num_shards_;

  string file_root_, file_list_;
  vector<std::pair<string, int>> image_label_pairs_;
  bool shuffle_after_epoch_;
  Index current_index_;
  int current_epoch_;
  FileStream::FileStreamMappinReserver mmap_reserver;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_FILE_LOADER_H_
