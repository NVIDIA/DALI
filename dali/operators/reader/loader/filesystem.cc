// Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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
#include <sys/stat.h>
#include <algorithm>
#include <string>
#include <cstring>
#include <utility>
#include <vector>
#include "dali/operators/reader/loader/filesystem.h"
#include "dali/operators/reader/loader/utils.h"
#include "dali/core/error_handling.h"

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



inline void assemble_file_list(std::vector<std::pair<std::string, int>>& file_label_pairs,
                               const std::string &path, const std::string &curr_entry, int label) {
  std::string curr_dir_path = path + dir_sep + curr_entry;
  DIR *dir = opendir(curr_dir_path.c_str());

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
    std::string rel_path = curr_entry + dir_sep + std::string{entry->d_name};
    if (HasKnownExtension(std::string(entry->d_name))) {
      file_label_pairs.emplace_back(rel_path, label);
    }
  }
  closedir(dir);
}

vector<std::pair<string, int>> traverse_directories(const std::string &file_root) {
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
    std::string full_path = join_path(file_root, entry_name);
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
    assemble_file_list(file_label_pairs, file_root, entry_name_list[dir_count], dir_count);
  }
  // sort file names as well
  std::sort(file_label_pairs.begin(), file_label_pairs.end());
  printf("read %lu files from %lu directories\n", file_label_pairs.size(), entry_name_list.size());

  closedir(dir);

  return file_label_pairs;
}


inline void assemble_file_list(std::vector<std::string>& file_list,
                               const std::string &path, const std::string &curr_entry,
                               const std::string &filter) {
  std::string curr_dir_path = path + dir_sep + curr_entry;
  DIR *dir = opendir(curr_dir_path.c_str());

  dirent *entry;

  if (filter.empty()) {
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
      std::string rel_path = curr_entry + dir_sep + std::string{entry->d_name};
      if (HasKnownExtension(std::string(entry->d_name))) {
         file_list.push_back(rel_path);
      }
    }
  } else {
    // use glob to do the file search
    glob_t pglob;
    std::string pattern = curr_dir_path + dir_sep + filter;
    if (glob(pattern.c_str(), GLOB_TILDE, NULL, &pglob) == 0) {
      // iterate through the matched files
      for (unsigned int count = 0; count < pglob.gl_pathc; ++count) {
        std::string match(pglob.gl_pathv[count]);
        std::string rel_path = curr_entry + dir_sep + match.substr(match.find_last_of(dir_sep)+1);
        file_list.push_back(rel_path);
      }
      // clean up
      globfree(&pglob);
    }
  }
  closedir(dir);
}


vector<std::string> traverse_directories(const std::string &file_root, const std::string &filter) {
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
    std::string full_path = join_path(file_root, entry_name);
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
    assemble_file_list(file_list, file_root, entry_name_list[dir_count], filter);
  }
  // sort file names as well
  std::sort(file_list.begin(), file_list.end());
  printf("read %lu files from %lu directories\n", file_list.size(), entry_name_list.size());

  closedir(dir);

  return file_list;
}

}  // namespace filesystem
}  // namespace dali
