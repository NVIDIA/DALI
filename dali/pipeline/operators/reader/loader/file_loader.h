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

#include "dali/common.h"
#include "dali/pipeline/operators/reader/loader/loader.h"
#include "dali/util/file.h"

namespace dali {

namespace filesystem {

void assemble_file_list(const std::string& path, int label,
                        std::vector<std::pair<std::string, int>> *file_label_pairs) {
  DIR *dir = opendir(path.c_str());
  struct dirent *entry;

  while ((entry = readdir(dir))) {
    std::string full_path = path + "/" + std::string{entry->d_name};
    struct stat s;
    stat(full_path.c_str(), &s);
    if (S_ISREG(s.st_mode)) {
      file_label_pairs->push_back(std::make_pair(full_path, label));
    }
  }
  closedir(dir);
}

vector<std::pair<string, int>> traverse_directories(const std::string& path) {
  // open the root
  DIR *dir = opendir(path.c_str());

  struct dirent *entry;
  int dir_count = 0;

  std::vector<std::pair<std::string, int>> file_label_pairs;

  while ((entry = readdir(dir))) {
    struct stat s;
    std::string full_path = path + "/" + std::string(entry->d_name);
    int ret = stat(full_path.c_str(), &s);
    DALI_ENFORCE(ret == 0,
        "Could not access " + full_path + " during directory traversal.");
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
    if (S_ISDIR(s.st_mode)) {
      assemble_file_list(full_path, dir_count, &file_label_pairs);
      dir_count++;
    }
  }
  printf("read %lu files from %d directories\n", file_label_pairs.size(), dir_count);

  closedir(dir);

  return file_label_pairs;
}

}  // namespace filesystem

class FileLoader : public Loader<CPUBackend> {
 public:
  explicit FileLoader(const OpSpec& spec)
    : Loader<CPUBackend>(spec),
      file_root_(spec.GetArgument<string>("file_root")),
      current_index_(0) {
    file_list_ = spec.GetArgument<string>("file_list");
    if (file_list_ == "") {
      image_label_pairs_ = filesystem::traverse_directories(file_root_);
    } else {
      // load (path, label) pairs from list
      std::ifstream s(file_list_);
      DALI_ENFORCE(s.is_open());

      string image_file;
      int label;
      while (s >> image_file >> label) {
        auto p = std::make_pair(file_root_ + "/" + image_file, label);
        image_label_pairs_.push_back(p);
      }
      DALI_ENFORCE(s.eof(), "Wrong format of file_list.");
    }

    DALI_ENFORCE(Size() > 0, "No files found.");
    // first / only shard: no change needed
    if (shard_id_ == 0) return;

    if (shuffle_) {
      // seeded with hardcoded value to get
      // the same sequence on every shard
      std::mt19937 g(524287);
      std::shuffle(image_label_pairs_.begin(), image_label_pairs_.end(), g);
    }

    int samples_per_shard = Size() / num_shards_;
    current_index_ = shard_id_ * samples_per_shard;
  }

  void ReadSample(Tensor<CPUBackend>* tensor) override {
    auto image_pair = image_label_pairs_[current_index_++];

    // handle wrap-around
    if (current_index_ == Size()) {
      current_index_ = 0;
    }

    FileStream *current_image = FileStream::Open(image_pair.first);
    Index image_size = current_image->Size();

    // resize tensor to hold [image, label]
    tensor->Resize({image_size + static_cast<Index>(sizeof(int))});

    // copy the image
    current_image->Read(tensor->mutable_data<uint8_t>(), image_size);

    // close the file handle
    current_image->Close();

    // copy the label
    *(reinterpret_cast<int*>(&tensor->mutable_data<uint8_t>()[image_size])) = image_pair.second;
  }

  Index Size() override {
    return static_cast<Index>(image_label_pairs_.size());
  }

 protected:
  using Loader<CPUBackend>::shard_id_;
  using Loader<CPUBackend>::num_shards_;

  string file_root_, file_list_;
  vector<std::pair<string, int>> image_label_pairs_;
  Index current_index_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_LOADER_FILE_LOADER_H_
