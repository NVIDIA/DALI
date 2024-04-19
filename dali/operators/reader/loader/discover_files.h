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

#ifndef DALI_OPERATORS_READER_LOADER_DISCOVER_FILES_H_
#define DALI_OPERATORS_READER_LOADER_DISCOVER_FILES_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "dali/core/common.h"

namespace dali {

inline bool starts_with(const std::string &str, const char *prefix) {
  // TODO(janton): this is a substitute for C++20's string::starts_with
  // trick: this only matches if the prefix is found at the beginning of the string
  return str.rfind(prefix, 0) == 0;
}

struct FileLabelEntry {
  std::string filename;
  // only if label_from_subdir==true
  std::optional<int> label = {};
  // only populated when size is known without opening (e.g. s3)
  std::optional<std::size_t> size = {};
};

struct FileDiscoveryOptions {
  bool label_from_subdir = true;  // if true, the directory is expected to contain a subdirectory
                                  // for each category. The traversal will assign ascending integers
                                  // as labels for each of those
  bool case_sensitive_filter = false;     // whether the filter patterns are case-sensitive
  std::vector<std::string> file_filters;  // pattern to apply to filenames
  std::vector<std::string> dir_filters;   // pattern to apply to subdirectories
};

/**
 * @brief Finds all (file, label, size) information, following the criteria given by opts.
 */
DLL_PUBLIC vector<FileLabelEntry> discover_files(const std::string &file_root,
                                                 const FileDiscoveryOptions &opts);

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_DISCOVER_FILES_H_
