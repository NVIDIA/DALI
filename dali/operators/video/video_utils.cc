// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/video/video_utils.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include "dali/operators/reader/loader/discover_files.h"
#include "dali/operators/reader/loader/filesystem.h"

namespace dali {

std::vector<VideoFileMeta> GetVideoFiles(const std::string& file_root,
                                         const std::vector<std::string>& filenames, bool use_labels,
                                         const std::vector<int>& labels,
                                         const std::string& file_list) {
  // open the root
  std::vector<VideoFileMeta> file_info;

  if (!file_root.empty()) {
    // Each subdirectory of file_root is a class; the files directly in file_root are skipped.
    // discover_files lists remote storage (s3://, gs://) as well as local directories.
    FileDiscoveryOptions opts;
    opts.label_from_subdir = true;
    opts.file_filters = {"*"};  // an empty filter list matches no local files
    auto entries = discover_files(file_root, opts);
    file_info.reserve(entries.size());
    for (auto &entry : entries) {
      file_info.push_back(
          VideoFileMeta{filesystem::join_path(file_root, entry.filename), *entry.label, 0, 0});
    }

    // discover_files returns the files grouped by directory - sort them by the full path
    std::sort(file_info.begin(), file_info.end());
  } else if (!file_list.empty()) {
    // load (path, label) pairs from list
    std::ifstream s(file_list);
    DALI_ENFORCE(s.is_open(), file_list + " could not be opened.");

    string line;
    string video_file;
    int label;
    float start;
    float end;
    int line_num = 0;
    while (std::getline(s, line)) {
      line_num++;
      video_file.clear();
      label = -1;
      start = end = 0;
      std::istringstream file_line(line);
      file_line >> video_file >> label;
      if (video_file.empty())
        continue;
      DALI_ENFORCE(label >= 0, "Label value should be >= 0 in file_list at line number: " +
                                   to_string(line_num) + ", filename: " + video_file);
      if (file_line >> start) {
        if (file_line >> end) {
          if (start == end) {
            DALI_WARN(
                "Start and end time/frame are the same, skipping the file, in file_list "
                "at line number: " +
                to_string(line_num) + ", filename: " + video_file);
            continue;
          }
        }
      }
      file_info.push_back(VideoFileMeta{video_file, label, start, end});
    }

    DALI_ENFORCE(s.eof(), "Wrong format of file_list.");
    s.close();
  } else {
    file_info.reserve(filenames.size());
    if (use_labels) {
      if (!labels.empty()) {
        for (size_t i = 0; i < filenames.size(); ++i) {
          file_info.push_back(VideoFileMeta{filenames[i], labels[i], 0, 0});
        }
      } else {
        for (size_t i = 0; i < filenames.size(); ++i) {
          file_info.push_back(VideoFileMeta{filenames[i], static_cast<int>(i), 0, 0});
        }
      }
    } else {
      for (size_t i = 0; i < filenames.size(); ++i) {
        file_info.push_back(VideoFileMeta{filenames[i], 0, 0, 0});
      }
    }
  }

  LOG_LINE << "read " << file_info.size() << " files\n";

  return file_info;
}

std::string av_error_string(int ret) {
  static char msg[AV_ERROR_MAX_STRING_SIZE];
  memset(msg, 0, sizeof(msg));
  return std::string(av_make_error_string(msg, AV_ERROR_MAX_STRING_SIZE, ret));
}

}  // namespace dali
