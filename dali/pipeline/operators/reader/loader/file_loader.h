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

#ifndef DALI_PIPELINE_OPERATORS_READER_LOADER_FILE_LOADER_H_
#define DALI_PIPELINE_OPERATORS_READER_LOADER_FILE_LOADER_H_

#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>

#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <algorithm>

#include "dali/common.h"
#include "dali/pipeline/operators/reader/loader/loader.h"
#include "dali/util/file.h"

namespace dali {

namespace filesystem {

vector<std::pair<string, int>> traverse_directories(const std::string& path);

}  // namespace filesystem

class FileLoader : public Loader<CPUBackend> {
 public:
  explicit inline FileLoader(const OpSpec& spec,
      vector<std::pair<string, int>> image_label_pairs = std::vector<std::pair<string, int>>())
    : Loader<CPUBackend>(spec),
      file_root_(spec.GetArgument<string>("file_root")),
      image_label_pairs_(image_label_pairs),
      current_index_(0) {
    file_list_ = spec.GetArgument<string>("file_list");

    if (image_label_pairs_.empty()) {
      if (file_list_ == "") {
        image_label_pairs_ = filesystem::traverse_directories(file_root_);
      } else {
        // load (path, label) pairs from list
        std::ifstream s(file_list_);
        DALI_ENFORCE(s.is_open());

        string image_file;
        int label;
        while (s >> image_file >> label) {
          auto p = std::make_pair(image_file, label);
          image_label_pairs_.push_back(p);
        }
        DALI_ENFORCE(s.eof(), "Wrong format of file_list.");
      }
    }

    DALI_ENFORCE(Size() > 0, "No files found.");

    if (shuffle_) {
      // seeded with hardcoded value to get
      // the same sequence on every shard
      std::mt19937 g(524287);
      std::shuffle(image_label_pairs_.begin(), image_label_pairs_.end(), g);
    }

    current_index_ = start_index(shard_id_, num_shards_, Size());
  }

  void ReadSample(Tensor<CPUBackend>* tensor);

  Index Size() override;

 protected:
  using Loader<CPUBackend>::shard_id_;
  using Loader<CPUBackend>::num_shards_;

  string file_root_, file_list_;
  vector<std::pair<string, int>> image_label_pairs_;
  Index current_index_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_LOADER_FILE_LOADER_H_
