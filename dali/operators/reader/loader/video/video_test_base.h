// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <vector>
#include <string>

#include "dali/test/dali_test_config.h"
#include "dali/test/cv_mat_utils.h"

namespace dali {
void CompareFrameAvgError(
  const uint8_t *ground_truth, const uint8_t *frame, int frame_size, double eps = 1.0);

class TestVideo {
 public:
  TestVideo(std::string folder_path, bool is_vfr):
    is_vfr_(is_vfr) {
    std::ifstream frames_list(folder_path + "frames_list.txt");
    std::string frame_filename;

    while (std::getline(frames_list, frame_filename)) {
      cv::Mat frame;
      cv::cvtColor(cv::imread(folder_path + frame_filename), frame, cv::COLOR_BGR2RGB);
      DALI_ENFORCE(frame.isContinuous(), "Loaded frame is not continuous in memory");

      frames_.push_back(frame);
    }
  }

  int NumFrames() const { return frames_.size(); }

  int NumChannels() const { return 3; }

  int Width() const { return frames_[0].cols; }

  int Height() const { return frames_[0].rows; }

  int FrameSize() const { return Height() * Width() * NumChannels(); }

  void CompareFrame(int frame_id, const uint8_t *frame, int eps = 10);

  void CompareFrameAvgError(int frame_id, const uint8_t *frame, double eps = 1.0);

  bool IsVfr() { return is_vfr_; }

  std::vector<cv::Mat> frames_;
  bool is_vfr_ = false;
};

class VideoTestBase : public ::testing::Test {
 public:
  int NumVideos() const { return cfr_videos_.size(); }

  int MaxFrameSize() const {
    return std::max(cfr_videos_[0].FrameSize(), cfr_videos_[1].FrameSize());
  }

  std::vector<char> MemoryVideo(const std::string &path) const;

  void RunFailureTest(std::function<void()> body, std::string expected_error);

 protected:
  static std::vector<std::string> cfr_videos_frames_paths_;
  static std::vector<std::string> vfr_videos_frames_paths_;
  static std::vector<std::string> vfr_hevc_videos_frames_paths_;

  static std::vector<std::string> cfr_videos_paths_;
  static std::vector<std::string> vfr_videos_paths_;

  static std::vector<std::string> cfr_hevc_videos_paths_;
  static std::vector<std::string> vfr_hevc_videos_paths_;

  static std::vector<std::string> cfr_mpeg4_videos_paths_;
  static std::vector<std::string> vfr_mpeg4_videos_paths_;

  static std::vector<std::string> cfr_mpeg4_mkv_videos_paths_;
  static std::vector<std::string> vfr_mpeg4_mkv_videos_paths_;

  static std::vector<TestVideo> cfr_videos_;
  static std::vector<TestVideo> vfr_videos_;
  static std::vector<TestVideo> vfr_hevc_videos_;

  static std::vector<std::string> cfr_raw_h264_videos_paths_;
  static std::vector<std::string> cfr_raw_h265_videos_paths_;

  static void SetUpTestSuite();
};
}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_TEST_BASE_H_
