// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_TEST_BASE_H_
#define DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_TEST_BASE_H_

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <vector>
#include <string>


namespace dali {
class VideoTestBase : public ::testing::Test {
 public:
  int NumVideos() const { return cfr_frames_.size(); }

  int NumFrames(int i) const { return cfr_frames_[i].size(); }

  int Channels() const { return 3; }

  int Width(int i) const { return cfr_frames_[i][0].cols; }

  int Height(int i) const { return cfr_frames_[i][0].rows; }

  int FrameSize(int i) const { return Height(i) * Width(i) * Channels(); }

  void CompareFrames(const uint8_t *frame, const uint8_t *gt, int size, int eps = 10);

  uint8_t *GetCfrFrame(int video_id, int frame_id) { return cfr_frames_[video_id][frame_id].data; }

  uint8_t *GetVfrFrame(int video_id, int frame_id) { return vfr_frames_[video_id][frame_id].data; }

 protected:
  static std::vector<std::vector<cv::Mat>> cfr_frames_;
  static std::vector<std::vector<cv::Mat>> vfr_frames_;

  static void SetUpTestSuite();
  static void LoadFrames(
    std::vector<std::string> &paths, std::vector<std::vector<cv::Mat>> &frames);
};
}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_TEST_BASE_H_
