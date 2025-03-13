// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <sys/stat.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

namespace dali {

inline void assemble_video_list(const std::string& path, const std::string& curr_entry, int label,
                                std::vector<VideoFileMeta>& file_info) {
  std::string curr_dir_path = path + "/" + curr_entry;
  DIR* dir = opendir(curr_dir_path.c_str());
  DALI_ENFORCE(dir != nullptr, "Directory " + curr_dir_path + " could not be opened");

  struct dirent* entry;

  while ((entry = readdir(dir))) {
    std::string full_path = curr_dir_path + "/" + std::string{entry->d_name};
#ifdef _DIRENT_HAVE_D_TYPE
    /*
     * Regular files and symlinks supported. If FS returns DT_UNKNOWN,
     * filename is validated.
     */
    if (entry->d_type != DT_REG && entry->d_type != DT_LNK && entry->d_type != DT_UNKNOWN) {
      continue;
    }
#endif
    file_info.push_back(VideoFileMeta{full_path, label, 0, 0});
  }
  closedir(dir);
}

std::vector<VideoFileMeta> GetVideoFiles(const std::string& file_root,
                                         const std::vector<std::string>& filenames, bool use_labels,
                                         const std::vector<int>& labels,
                                         const std::string& file_list) {
  // open the root
  std::vector<VideoFileMeta> file_info;
  std::vector<std::string> entry_name_list;

  if (!file_root.empty()) {
    DIR* dir = opendir(file_root.c_str());

    DALI_ENFORCE(dir != nullptr, "Directory " + file_root + " could not be opened.");

    struct dirent* entry;

    while ((entry = readdir(dir))) {
      struct stat s;
      std::string entry_name(entry->d_name);
      std::string full_path = file_root + "/" + entry_name;
      int ret = stat(full_path.c_str(), &s);
      DALI_ENFORCE(ret == 0, "Could not access " + full_path + " during directory traversal.");
      if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
        continue;
      if (S_ISDIR(s.st_mode)) {
        entry_name_list.push_back(entry_name);
      }
    }
    closedir(dir);
    // sort directories to preserve class alphabetic order, as readdir could
    // return unordered dir list. Otherwise file reader for training and validation
    // could return directories with the same names in completely different order
    std::sort(entry_name_list.begin(), entry_name_list.end());
    for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
      assemble_video_list(file_root, entry_name_list[dir_count], dir_count, file_info);
    }

    // sort file names as well
    std::sort(file_info.begin(), file_info.end());
  } else if (!file_list.empty()) {
    // load (path, label) pairs from list
    std::ifstream s(file_list);
    DALI_ENFORCE(s.is_open(), file_list + " could not be opened.");

    string line;
    string video_file;
    int label;
    float start_time;
    float end_time;
    int line_num = 0;
    while (std::getline(s, line)) {
      line_num++;
      video_file.clear();
      label = -1;
      start_time = end_time = 0;
      std::istringstream file_line(line);
      file_line >> video_file >> label;
      if (video_file.empty())
        continue;
      DALI_ENFORCE(label >= 0, "Label value should be >= 0 in file_list at line number: " +
                                   to_string(line_num) + ", filename: " + video_file);
      if (file_line >> start_time) {
        if (file_line >> end_time) {
          if (start_time == end_time) {
            DALI_WARN(
                "Start and end time/frame are the same, skipping the file, in file_list "
                "at line number: " +
                to_string(line_num) + ", filename: " + video_file);
            continue;
          }
        }
      }
      file_info.push_back(VideoFileMeta{video_file, label, start_time, end_time});
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

  LOG_LINE << "read " << file_info.size() << " files from " << entry_name_list.size()
           << " directories\n";

  return file_info;
}

std::string av_error_string(int ret) {
  static char msg[AV_ERROR_MAX_STRING_SIZE];
  memset(msg, 0, sizeof(msg));
  return std::string(av_make_error_string(msg, AV_ERROR_MAX_STRING_SIZE, ret));
}

}  // namespace dali
