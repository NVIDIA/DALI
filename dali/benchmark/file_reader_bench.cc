// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <benchmark/benchmark.h>
#include <dirent.h>
#include <fnmatch.h>
#include <sys/stat.h>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/operators/reader/loader/filesystem.h"
#include "dali/operators/reader/loader/utils.h"
#include "dali/test/dali_test_config.h"

namespace dali {

static std::vector<std::string> getSubdirs(const std::string &file_root) {
  // open the root
  DIR *dir = opendir(file_root.c_str());

  DALI_ENFORCE(dir != nullptr, make_string("Directory ", file_root, " could not be opened."));

  struct dirent *entry;

  std::vector<std::string> file_list;
  std::vector<std::string> entry_name_list;

  // always append the root current directory
  entry_name_list.push_back(".");

  // now traverse sub-directories
  while ((entry = readdir(dir))) {
    struct stat s;
    std::string entry_name(entry->d_name);
    std::string full_path = filesystem::join_path(file_root, entry_name);
    int ret = stat(full_path.c_str(), &s);
    DALI_ENFORCE(ret == 0,
                 make_string("Could not access ", full_path, " during directory traversal."));
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
      continue;
    if (S_ISDIR(s.st_mode)) {
      entry_name_list.push_back(std::move(entry_name));
    }
  }

  // sort directories
  std::sort(entry_name_list.begin(), entry_name_list.end());

  closedir(dir);
  return entry_name_list;
}

static void BM_TraverseGlob(benchmark::State &state) {
  std::vector<std::string> filters{"*.jpg", "*.jpeg", "*.png"};
  std::vector<std::string> file_list;
  const std::string file_root(testing::dali_extra_path() + "/db/single/mixed");
  auto paths = getSubdirs(file_root);

  for (auto _ : state) {
    for (auto &curr_entry : paths) {
      std::string curr_dir_path = file_root + filesystem::dir_sep + curr_entry;
      DIR *dir = opendir(curr_dir_path.c_str());

      dirent *entry;

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
        std::string rel_path = curr_entry + filesystem::dir_sep + std::string{entry->d_name};
        for (auto &filter : filters) {
          if (fnmatch(filter.c_str(), entry->d_name, 0) == 0) {
            file_list.emplace_back(rel_path);
            continue;
          }
        }
      }
      closedir(dir);
    }
  }
}
// Register the function as a benchmark
BENCHMARK(BM_TraverseGlob);

// Define another benchmark
static void BM_TraverseRfind(benchmark::State &state) {
  std::vector<std::string> file_list;
  const std::string file_root(testing::dali_extra_path() + "/db/single/mixed");
  auto paths = getSubdirs(file_root);

  for (auto _ : state) {
    for (auto &curr_entry : paths) {
      std::string curr_dir_path = file_root + filesystem::dir_sep + curr_entry;
      DIR *dir = opendir(curr_dir_path.c_str());

      dirent *entry;

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
        if (HasKnownExtension(std::string(entry->d_name))) {
          std::string rel_path = curr_entry + filesystem::dir_sep + std::string{entry->d_name};
          file_list.emplace_back(rel_path);
        }
      }
      closedir(dir);
    }
  }
}
BENCHMARK(BM_TraverseRfind);

}  // namespace dali
