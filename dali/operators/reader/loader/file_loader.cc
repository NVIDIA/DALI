// Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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
#include <glob.h>
#include <memory>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/file_loader.h"
#include "dali/util/file.h"
#include "dali/operators/reader/loader/utils.h"

namespace dali {


inline void assemble_file_list(const std::string& path, const std::string& curr_entry,
                               const std::string& filter, std::vector<std::string>* file_list_) {
  std::string curr_dir_path = path + "/" + curr_entry;
  DIR *dir = opendir(curr_dir_path.c_str());

  struct dirent *entry;

  if (filter == "") {
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
      if (HasKnownExtension(std::string(entry->d_name))) {
         file_list_->push_back(rel_path);
      }
    }
  } else {
    // use glob to do the file search
    glob_t pglob;
    std::string pattern = curr_dir_path + '/' + filter;
    glob(pattern.c_str(), GLOB_TILDE, NULL, &pglob);

    // iterate through the matched files
    for (unsigned int count = 0; count < pglob.gl_pathc; ++count) {
      std::string match(pglob.gl_pathv[count]);
      std::string rel_path = curr_entry + "/" + match.substr(match.find_last_of("/")+1);
      file_list_->push_back(rel_path);
    }
    // clean up
    globfree(&pglob);
  }
  closedir(dir);
}


vector<std::string> filesystem::traverse_directories(const std::string& file_root,
                                                     std::string filter) {
  // open the root
  DIR *dir = opendir(file_root.c_str());

  DALI_ENFORCE(dir != nullptr,
      "Directory " + file_root + " could not be opened.");

  struct dirent *entry;

  std::vector<std::string> file_list;
  std::vector<std::string> entry_name_list;

  // always append the root current directory
  entry_name_list.push_back(".");

  // now traverse sub-directories
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

  // sort directories
  std::sort(entry_name_list.begin(), entry_name_list.end());
  for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
    assemble_file_list(file_root, entry_name_list[dir_count], filter, &file_list);
  }
  // sort file names as well
  std::sort(file_list.begin(), file_list.end());
  printf("read %lu files from %lu directories\n", file_list.size(), entry_name_list.size());

  closedir(dir);

  return file_list;
}

void FileLoader::PrepareEmpty(ImageFileWrapper &image_file) {
  PrepareEmptyTensor(image_file.image);
  image_file.filename = "";
}

void FileLoader::ReadSample(ImageFileWrapper& imfile) {
  auto image_file = images_[current_index_++];

  // handle wrap-around
  MoveToNextShard(current_index_);

  // metadata info
  DALIMeta meta;
  meta.SetSourceInfo(image_file);
  meta.SetSkipSample(false);

  // if image is cached, skip loading
  if (ShouldSkipImage(image_file)) {
    meta.SetSkipSample(true);
    imfile.image.Reset();
    imfile.image.SetMeta(meta);
    imfile.image.set_type(TypeInfo::Create<uint8_t>());
    imfile.image.Resize({0});
    imfile.filename = "";
    return;
  }

  auto current_image = FileStream::Open(file_root_ + "/" + image_file, read_ahead_);
  Index image_size = current_image->Size();

  if (copy_read_data_) {
    if (imfile.image.shares_data()) {
      imfile.image.Reset();
    }
    imfile.image.Resize({image_size});
    // copy the image
    current_image->Read(imfile.image.mutable_data<uint8_t>(), image_size);
  } else {
    auto p = current_image->Get(image_size);
    // Wrap the raw data in the Tensor object.
    imfile.image.ShareData(p, image_size, {image_size});
    imfile.image.set_type(TypeInfo::Create<uint8_t>());
  }

  // close the file handle
  current_image->Close();

  // set metadata
  imfile.image.SetMeta(meta);

  // set string
  imfile.filename = file_root_ + "/" + image_file;
}

Index FileLoader::SizeImpl() {
  return static_cast<Index>(images_.size());
}
}  // namespace dali
