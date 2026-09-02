// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/reader/loader/discover_files_gcs.h"
#include <fnmatch.h>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>
#include "dali/operators/reader/loader/discover_files.h"
#include "dali/util/gcs_client_manager.h"
#include "dali/util/gcs_filesystem.h"

namespace dali {

// We are using std::filesystem to analyze URI relative paths, which wouldn't be OK in non-UNIX
// based systems
#ifndef __unix__
#error This code works only in UNIX-based systems
#endif

std::vector<FileLabelEntry> gcs_discover_files(const std::string &file_root,
                                               const FileDiscoveryOptions &opts) {
  assert(starts_with(file_root, "gs://"));
  auto gcs_object_location = gcs_filesystem::parse_uri(file_root);
  std::filesystem::path parent_object_key(gcs_object_location.object);
  auto count_elems = [](const std::filesystem::path &p) {
    size_t k = 0;
    for (auto &elem : p)
      k++;
    return k;
  };
  std::vector<FileLabelEntry> entries;
  // in case that files are not visited in lexicographical order, we remember previously assigned
  // labels
  std::unordered_map<std::string, int> labels;
  int next_label = 0;  // next free-label to be assigned
  auto client = GCSClientManager::Instance().client();
  gcs_filesystem::list_objects_f(
      client, gcs_object_location, [&](const std::string &object_key, size_t object_size) {
        auto p = std::filesystem::relative(object_key, parent_object_key);
        auto path_elems = count_elems(p);
        // We only look at one subdir level. Fewer than two components means either an object
        // directly under the listed prefix, or the prefix's own directory marker, which relative()
        // maps to "."; neither is a labelled file, and both must be rejected before dereferencing
        // the second component below.
        if (path_elems != 2)
          return;
        const auto& subdir = p.begin()->native();
        const auto& fname = (++p.begin())->native();
        // GCS directory markers are zero-byte objects whose name ends with '/'. A trailing
        // separator becomes an empty final component, so "<prefix>/class/" arrives here as
        // ("class", "") - a directory, not a file.
        if (fname.empty())
          return;
        bool subdir_ok = opts.dir_filters.empty();
        bool fname_ok = opts.file_filters.empty();
        for (auto &filter : opts.dir_filters) {
          if (fnmatch(filter.c_str(), subdir.c_str(),
                      opts.case_sensitive_filter ? 0 : FNM_CASEFOLD) == 0) {
            subdir_ok |= true;
            break;
          }
        }

        for (auto &filter : opts.file_filters) {
          if (fnmatch(filter.c_str(), fname.c_str(),
                      opts.case_sensitive_filter ? 0 : FNM_CASEFOLD) == 0) {
            fname_ok |= true;
            break;
          }
        }

        if (!subdir_ok || !fname_ok)
          return;

        if (opts.label_from_subdir) {
          int curr_label = -1;
          auto it = labels.find(subdir);
          if (it == labels.end()) {
            curr_label = labels[subdir] = next_label++;
          } else {
            curr_label = it->second;
          }
          entries.push_back({p, curr_label, object_size});
        } else {
          entries.push_back({p, std::nullopt, object_size});
        }
      });
  return entries;
}

}  // namespace dali
