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

#ifndef DALI_OPERATORS_VIDEO_VIDEO_READER_UTILS_H_
#define DALI_OPERATORS_VIDEO_VIDEO_READER_UTILS_H_

#include <dirent.h>
#include <string>
#include <vector>
#include "dali/core/error_handling.h"
#include "libavutil/rational.h"

namespace dali {

struct VideoFileMeta {
  std::string video_file;
  int label;
  float start_time;
  float end_time;
  bool operator<(const VideoFileMeta& right) {
    return video_file < right.video_file;
  }
};

inline float TimestampToSeconds(AVRational timebase, int64_t timestamp) {
  return static_cast<float>(
    static_cast<double>(timestamp) * timebase.num / timebase.den);
}

inline int64_t SecondsToTimestamp(AVRational timebase, float seconds) {
  return static_cast<int64_t>(
    static_cast<double>(seconds) * timebase.den / timebase.num);
}

std::vector<VideoFileMeta> GetVideoFiles(const std::string& file_root,
                                         const std::vector<std::string>& filenames, bool use_labels,
                                         const std::vector<int>& labels,
                                         const std::string& file_list);

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_VIDEO_READER_UTILS_H_
