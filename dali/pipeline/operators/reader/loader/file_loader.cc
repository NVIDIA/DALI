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
#include <errno.h>
#include <memory>

#include "dali/common.h"
#include "dali/image/image.h"
#include "dali/pipeline/operators/reader/loader/file_loader.h"
#include "dali/util/file.h"

namespace dali {

inline void assemble_file_list(const std::string& path, const std::string& curr_entry, int label,
                        std::vector<std::pair<std::string, int>> *file_label_pairs) {
  std::string curr_dir_path = path + "/" + curr_entry;
  DIR *dir = opendir(curr_dir_path.c_str());

  struct dirent *entry;

  while ((entry = readdir(dir))) {
    std::string full_path = curr_dir_path + "/" + std::string{entry->d_name};
#ifdef _DIRENT_HAVE_D_TYPE
    /*
     * we support only regular files and symlinks, if FS returns DT_UNKNOWN
     * it doesn't mean anything and let us validate filename itself
     */
    if (entry->d_type != DT_REG && entry->d_type != DT_LNK &&
        entry->d_type != DT_UNKNOWN) {
      continue;
    }
#endif
    std::string rel_path = curr_entry + "/" + std::string{entry->d_name};
    if (HasKnownImageExtension(std::string(entry->d_name))) {
      file_label_pairs->push_back(std::make_pair(rel_path, label));
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
  // sort file names as well
  std::sort(file_label_pairs.begin(), file_label_pairs.end());
  printf("read %lu files from %lu directories\n", file_label_pairs.size(), entry_name_list.size());

  closedir(dir);

  return file_label_pairs;
}

void FileLoader::PrepareEmpty(ImageLabelWrapper &image_label) {
  PrepareEmptyTensor(image_label.image);
}

void FileLoader::ReadSample(ImageLabelWrapper &image_label) {
  auto image_pair = image_label_pairs_[current_index_++];

  // handle wrap-around
  MoveToNextShard(current_index_);

  // copy the label
  image_label.label = image_pair.second;
  image_label.image.SetSourceInfo(image_pair.first);
  image_label.image.SetSkipSample(false);

  // if image is cached, skip loading
  if (ShouldSkipImage(image_pair.first)) {
    image_label.image.set_type(TypeInfo::Create<uint8_t>());
    image_label.image.Resize({1});
    image_label.image.SetSkipSample(true);
    return;
  }

  auto current_image = FileStream::Open(file_root_ + "/" + image_pair.first, read_ahead_);
  Index image_size = current_image->Size();

  if (copy_read_data_) {
    image_label.image.Resize({image_size});
    // copy the image
    current_image->Read(image_label.image.mutable_data<uint8_t>(), image_size);
  } else {
    auto p = current_image->Get(image_size);
    // Wrap the raw data in the Tensor object.
    image_label.image.ShareData(p, image_size, {image_size});
    image_label.image.set_type(TypeInfo::Create<uint8_t>());
  }

  image_label.image.SetSourceInfo(image_pair.first);
  // close the file handle
  current_image->Close();

  // copy the label
  image_label.label = image_pair.second;
}

Index FileLoader::SizeImpl() {
  return static_cast<Index>(image_label_pairs_.size());
}
}  // namespace dali
