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

#include "dali/operators/reader/loader/discover_files.h"
#include <dirent.h>
#include <errno.h>
#include <fnmatch.h>
#include <glob.h>
#include <sys/stat.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "dali/core/call_at_exit.h"
#include "dali/core/error_handling.h"
#include "dali/operators/reader/loader/filesystem.h"
#include "dali/operators/reader/loader/utils.h"
#if AWSSDK_ENABLED
#include "dali/operators/reader/loader/discover_files_s3.h"
#endif

namespace dali {

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
    std::string full_path = filesystem::join_path(parent_dir, entry_name);
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

std::vector<FileLabelEntry> discover_files(const std::string &file_root,
                                           const FileDiscoveryOptions &opts) {
  bool is_s3 = starts_with(file_root, "s3://");
  if (is_s3) {
#if AWSSDK_ENABLED
    return s3_discover_files(file_root, opts);
#else
    DALI_FAIL("This version of DALI was not built with AWS S3 storage support.");
#endif
  }

  std::vector<std::string> subdirs;
  subdirs = list_subdirectories(file_root, opts.dir_filters, opts.case_sensitive_filter);
  std::vector<FileLabelEntry> entries;
  auto process_dir = [&](const std::string &rel_dirpath, std::optional<int> label = {}) {
    auto full_dirpath = filesystem::join_path(file_root, rel_dirpath);
    auto tmp_files = list_files(full_dirpath, opts.file_filters, opts.case_sensitive_filter);
    for (const auto &f : tmp_files) {
      entries.push_back({filesystem::join_path(rel_dirpath, f), label});
    }
  };

  // if we are in "label_from_subdir" mode, we need a subdir to infer the label, therefore we don't
  // visit the current directory
  if (!opts.label_from_subdir) {
    process_dir(".");
  }
  for (unsigned dir_idx = 0; dir_idx < subdirs.size(); ++dir_idx) {
    process_dir(subdirs[dir_idx],
                opts.label_from_subdir ? std::optional<int>{dir_idx} : std::nullopt);
  }
  size_t total_dir_count = opts.label_from_subdir ? subdirs.size() : subdirs.size() + 1;
  LOG_LINE << "read " << entries.size() << " files from " << total_dir_count << "directories\n";
  return entries;
}

}  // namespace dali
