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

#include "dali/operators/reader/loader/filesystem.h"
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
#include "dali/core/call_at_exit.h"
#include "dali/core/error_handling.h"
#include "dali/operators/reader/loader/utils.h"

namespace dali {
namespace filesystem {

inline bool starts_with(const std::string &str, const char *prefix) {
  // TODO(janton): this is a substitute for C++20's string::starts_with
  // trick: this only matches if the prefix is found at the beginning of the string
  return str.rfind(prefix, 0) == 0;
}

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
  if (dir[dir.length() - 1] == dir_sep)
    return dir + path;
  else
    return dir + dir_sep + path;
}

std::vector<std::string> list_subdirectories(const std::string &parent_dir,
                                             const std::vector<std::string> dir_filters = {},
                                             bool case_sensitive_filter = true) {
  // open the root
  DIR *dir = opendir(parent_dir.c_str());
  DALI_ENFORCE(dir != nullptr, make_string("Failed to open ", parent_dir));
  auto cleanup = AtScopeExit([&dir] {
    closedir(dir);
  });

  struct dirent *entry;
  std::vector<std::string> subdirs;

  while ((entry = readdir(dir))) {
    struct stat s;
    std::string entry_name(entry->d_name);
    std::string full_path = join_path(parent_dir, entry_name);
    int ret = stat(full_path.c_str(), &s);
    DALI_ENFORCE(ret == 0, "Could not access " + full_path + " during directory traversal.");
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
      continue;
    if (S_ISDIR(s.st_mode)) {
      if (dir_filters.empty()) {
        subdirs.push_back(entry_name);
      } else {
        for (auto &filter : dir_filters) {
          if (fnmatch(filter.c_str(), entry_name.c_str(),
                      case_sensitive_filter ? 0 : FNM_CASEFOLD) == 0) {
            subdirs.push_back(entry_name);
          }
        }
      }
    }
  }
  // sort directories to preserve class alphabetic order, as readdir could
  // return unordered dir list. Otherwise file reader for training and validation
  // could return directories with the same names in completely different order
  std::sort(subdirs.begin(), subdirs.end());
  return subdirs;
}

std::vector<std::string> list_files(const std::string &parent_dir,
                                    const std::vector<std::string> filters = {},
                                    bool case_sensitive_filter = true) {
  DIR *dir = opendir(parent_dir.c_str());
  DALI_ENFORCE(dir != nullptr, make_string("Failed to open ", parent_dir));
  auto cleanup = AtScopeExit([&dir] {
    closedir(dir);
  });

  dirent *entry;
  std::vector<std::string> files;
  while ((entry = readdir(dir))) {
#ifdef _DIRENT_HAVE_D_TYPE
    /*
     * we support only regular files and symlinks, if FS returns DT_UNKNOWN
     * it doesn't mean anything and let us validate filename itself
     */
    if (entry->d_type != DT_REG && entry->d_type != DT_LNK && entry->d_type != DT_UNKNOWN) {
      continue;
    }
#endif
    std::string fname(entry->d_name);
    for (auto &filter : filters) {
      if (fnmatch(filter.c_str(), fname.c_str(), case_sensitive_filter ? 0 : FNM_CASEFOLD) == 0) {
        files.push_back(fname);
        break;
      }
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

vector<std::pair<string, int>> traverse_directories(const std::string &file_root,
                                                    const std::vector<std::string> &filters,
                                                    bool case_sensitive_filter,
                                                    const std::vector<std::string> &dir_filters) {
  std::vector<std::string> subdirs;
  bool is_s3 = starts_with(file_root, "s3://");
  if (is_s3)
    DALI_FAIL("This version of DALI was not built with AWS S3 storage support.")
  subdirs = list_subdirectories(file_root, dir_filters, case_sensitive_filter);

  std::vector<std::pair<std::string, int>> file_label_pairs;
  for (unsigned dir_count = 0; dir_count < subdirs.size(); ++dir_count) {
    std::vector<std::string> tmp_files;
    const auto &rel_dirpath = subdirs[dir_count];
    auto full_dirpath = join_path(file_root, rel_dirpath);
    tmp_files = list_files(full_dirpath, filters, case_sensitive_filter);
    for (const auto &f : tmp_files) {
      file_label_pairs.push_back({join_path(rel_dirpath, f), dir_count});
    }
  }
  LOG_LINE << "read " << file_label_pairs.size() << " files from " << subdirs.size()
           << "directories\n";
  return file_label_pairs;
}


vector<std::string> traverse_directories(const std::string &file_root, const std::string &filter) {
  std::vector<std::string> subdirs;
  bool is_s3 = starts_with(file_root, "s3://");
  if (is_s3)
    DALI_FAIL("This version of DALI was not built with AWS S3 storage support.");
  subdirs = list_subdirectories(file_root);

  std::vector<std::string> files;
  auto process_dir = [&](const std::string &rel_dirpath) {
    auto full_dirpath = join_path(file_root, rel_dirpath);
    auto tmp_files = list_files(full_dirpath, {filter}, true);
    for (const auto &f : tmp_files) {
      files.push_back(join_path(rel_dirpath, f));
    }
  };

  process_dir(".");  // process current dir as well
  for (const auto &subdir : subdirs)
    process_dir(subdir);

  LOG_LINE << "read " << files.size() << " files from " << subdirs.size() << "directories\n";
  return files;
}

}  // namespace filesystem
}  // namespace dali
