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

#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>

#include "dali/common.h"
#include "dali/pipeline/operators/reader/loader/file_loader.h"
#include "dali/util/file.h"

namespace dali {

inline void assemble_file_list(const std::string& path, const std::string& curr_entry, int label,
                        std::vector<std::pair<std::string, int>> *file_label_pairs) {
  std::string curr_dir_path = path + "/" + curr_entry;
  DIR *dir = opendir(curr_dir_path.c_str());

  struct dirent *entry;

  const std::vector<std::string> valid_extensions({".jpg", ".jpeg", ".png", ".bmp"});

  while ((entry = readdir(dir))) {
    std::string full_path = curr_dir_path + "/" + std::string{entry->d_name};
    struct stat s;
    stat(full_path.c_str(), &s);
    if (S_ISREG(s.st_mode)) {
      std::string rel_path = curr_entry + "/" + std::string{entry->d_name};
      std::string file_name_lowercase = std::string{entry->d_name};
      std::transform(file_name_lowercase.begin(), file_name_lowercase.end(),
                     file_name_lowercase.begin(), ::tolower);
      for (const std::string& s : valid_extensions) {
        size_t pos = file_name_lowercase.rfind(s);
        if (pos != std::string::npos && pos + s.size() == file_name_lowercase.size()) {
          file_label_pairs->push_back(std::make_pair(rel_path, label));
          break;
        }
      }
    }
  }
  closedir(dir);
}

vector<std::pair<string, int>> filesystem::traverse_directories(const std::string& file_root) {
  // open the root
  DIR *dir = opendir(file_root.c_str());

  DALI_ENFORCE(dir != nullptr,
      "Directory " + file_root + " could not be opened.");

  struct dirent *entry;

  std::vector<std::pair<std::string, int>> file_label_pairs;
  std::vector<std::string> entry_name_list;

  while ((entry = readdir(dir))) {
    struct stat s;
    std::string entry_name(entry->d_name);
    std::string full_path = file_root + "/" + entry_name;
    int ret = stat(full_path.c_str(), &s);
    DALI_ENFORCE(ret == 0,
        "Could not access " + full_path + " during directory traversal.");
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
    if (S_ISDIR(s.st_mode)) {
      entry_name_list.push_back(entry_name);
    }
  }
  // sort directories to preserve class alphabetic order, as readdir could
  // return unordered dir list. Otherwise file reader for training and validation
  // could return directories with the same names in completely different order
  std::sort(entry_name_list.begin(), entry_name_list.end());
  for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
      assemble_file_list(file_root, entry_name_list[dir_count], dir_count, &file_label_pairs);
  }
  printf("read %lu files from %lu directories\n", file_label_pairs.size(), entry_name_list.size());

  closedir(dir);

  return file_label_pairs;
}
void FileLoader::ReadSample(Tensor<CPUBackend>* tensor) {
  auto image_pair = image_label_pairs_[current_index_++];

  // handle wrap-around
  if (current_index_ == Size()) {
    current_index_ = 0;
  }

  FileStream *current_image = FileStream::Open(file_root_ + "/" + image_pair.first);
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

Index FileLoader::Size() {
  return static_cast<Index>(image_label_pairs_.size());
}
}  // namespace dali
