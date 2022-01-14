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

  void CompareFramesAvgError(const uint8_t *frame, const uint8_t *gt, int size, double eps = 1.0);

  uint8_t *GetCfrFrame(int video_id, int frame_id) { return cfr_frames_[video_id][frame_id].data; }

  uint8_t *GetVfrFrame(int video_id, int frame_id) { return vfr_frames_[video_id][frame_id].data; }

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
   * '/tmp/batch_002_sample_001_frame_000.png
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
    std::string folder_path,
    int width,
    int height);

 protected:
  static std::vector<std::vector<cv::Mat>> cfr_frames_;
  static std::vector<std::vector<cv::Mat>> vfr_frames_;

  static void SetUpTestSuite();
  static void LoadFrames(
    std::vector<std::string> &paths, std::vector<std::vector<cv::Mat>> &frames);
};
}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_VIDEO_VIDEO_TEST_BASE_H_
