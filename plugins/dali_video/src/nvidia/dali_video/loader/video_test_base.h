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

#pragma once

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <vector>
#include <string>

#include "dali/test/dali_test_config.h"
#include "dali/test/cv_mat_utils.h"

namespace dali_video {
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

  /**
   * @brief Utility to save decoded frame as a PNG file.
   * Frame is saved to the folder given as an argument.
   * Output file name is created with provaided ids of frame, sample and batch.
   *
   * For example:
   *
   * SaveFrame(ptr, 0, 1, 2, '/tmp', 800, 600)
   *
   * will save the frame as:
   *
   * /tmp/batch_002_sample_001_frame_000.png
   *
   * @param frame Frame data
   * @param frame_id FrameId that will be included in output file name
   * @param sample_id SampleId that will be included in output file name
   * @param batch_id BatchId that will be included in output file name
   * @param folder_path Path to a destination folder
   * @param width Frame width in pixels
   * @param height Frame height in pixels
   */
  void SaveFrame(
    uint8_t *frame,
    int frame_id,
    int sample_id,
    int batch_id,
    const std::string &folder_path,
    int width,
    int height);

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

  void RunFailureTest(std::function<void()> body, std::string expected_error);
};

}  // namespace dali_video
