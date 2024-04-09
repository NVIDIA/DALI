// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <fnmatch.h>
#include <glob.h>
#include <sys/stat.h>
#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/error_handling.h"
#include "dali/operators/reader/loader/filesystem.h"
#include "dali/operators/reader/loader/utils.h"

namespace dali {
namespace filesystem {

std::string join_path(const std::string &dir, const std::string &path) {
  if (dir.empty())
    return path;
  if (path.empty())
    return dir;
  if (path[0] == dir_sep)  // absolute path
    return path;
#ifdef WINVER
  if (path[1] == ':')
    return path;
#endif
  if (dir[dir.length()-1] == dir_sep)
    return dir + path;
  else
    return dir + dir_sep + path;
}

bool list_subdirectories(std::vector<std::string>& subdirs, const std::string &parent_dir) {
  // open the root
  DIR *dir = opendir(parent_dir.c_str());
  if (dir == nullptr)
    return false;

  struct dirent *entry;
  subdirs.clear();

  while ((entry = readdir(dir))) {
    struct stat s;
    std::string entry_name(entry->d_name);
    std::string full_path = join_path(parent_dir, entry_name);
    int ret = stat(full_path.c_str(), &s);
    DALI_ENFORCE(ret == 0,
        "Could not access " + full_path + " during directory traversal.");
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
    if (S_ISDIR(s.st_mode)) {
      subdirs.push_back(entry_name);
    }
  }
  // sort directories to preserve class alphabetic order, as readdir could
  // return unordered dir list. Otherwise file reader for training and validation
  // could return directories with the same names in completely different order
  std::sort(subdirs.begin(), subdirs.end());
  return true;
}


bool list_files(std::vector<std::string>& files, const std::string& parent_dir, 
                std::vector<std::string> filters = {}, bool case_sensitive_filter = true) {

  DIR *dir = opendir(parent_dir.c_str());
  if (dir == nullptr)
    return false;

  dirent *entry;

  while ((entry = readdir(dir))) {
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
    std::string fname(entry->d_name);
    for (auto &filter : filters) {
      if (fnmatch(filter.c_str(), fname.c_str(), case_sensitive_filter ? 0 : FNM_CASEFOLD) == 0) {
        files.emplace_back(fname);
        break;
      }
    }
  }
  closedir(dir);
  return true;
}

vector<std::pair<string, int>> traverse_directories(const std::string &file_root,
                                                    const std::vector<std::string> &filters,
                                                    const bool case_sensitive_filter) {
  // open the root
  std::vector<std::string> subdirs;
  bool ret = list_subdirectories(subdirs, file_root);
  DALI_ENFORCE(ret, "Directory " + file_root + " could not be opened.");

  std::vector<std::pair<std::string, int>> file_label_pairs;
  std::vector<std::string> files;
  for (unsigned dir_count = 0; dir_count < subdirs.size(); ++dir_count) {
    files.clear();
    auto dirpath = join_path(file_root, subdirs[dir_count]);
    list_files(files, dirpath, filters, case_sensitive_filter);
    for (const auto& f : files) {
      file_label_pairs.push_back({join_path(dirpath, f), dir_count});
    }
  }
  // sort file names as well
  std::sort(file_label_pairs.begin(), file_label_pairs.end());
  LOG_LINE  << "read " << file_label_pairs.size() << " files from " << subdirs.size()
            << "directories\n";
  return file_label_pairs;
}


vector<std::string> traverse_directories(const std::string &file_root, const std::string &filter) {

  // open the root
  std::vector<std::string> subdirs;
  bool ret = list_subdirectories(subdirs, file_root);
  subdirs.push_back(".");
  DALI_ENFORCE(ret, "Directory " + file_root + " could not be opened.");

  std::vector<std::string> files;
  std::vector<std::string> rel_path_files;
  for (unsigned dir_count = 0; dir_count < subdirs.size(); ++dir_count) {
    files.clear();
    auto dirpath = join_path(file_root, subdirs[dir_count]);
    list_files(rel_path_files, dirpath, {filter}, true);
    for (const auto& f : rel_path_files) {
      files.push_back(join_path(dirpath, f));
    }
  }
  // sort file names as well
  std::sort(files.begin(), files.end());
  LOG_LINE  << "read " << files.size() << " files from " << subdirs.size()
            << "directories\n";
  return files;
}

}  // namespace filesystem
}  // namespace dali
