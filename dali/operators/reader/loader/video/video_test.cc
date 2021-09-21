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

#include "dali/operators/reader/loader/video/video_test.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "dali/test/dali_test_config.h"
#include "dali/test/cv_mat_utils.h"

namespace dali {
VideoTest::VideoTest() {
  std::vector<std::string> cfr_frames_paths{
      testing::dali_extra_path() + "/db/video/cfr/frames_1/",
      testing::dali_extra_path() + "/db/video/cfr/frames_2/"};

  std::vector<std::string> vfr_frames_paths{
      testing::dali_extra_path() + "/db/video/vfr/frames_1/",
      testing::dali_extra_path() + "/db/video/vfr/frames_2/"};

  
  LoadFrames(cfr_frames_paths, cfr_frames_);
  LoadFrames(vfr_frames_paths, vfr_frames_);
}

void VideoTest::LoadFrames(std::vector<std::string> &paths, std::vector<std::vector<cv::Mat>> &out_frames) {
  for (auto &frames_path : paths) {
    char id_str[4];
    std::vector<cv::Mat> frames;

    int frame_id = 0;
    while (true) {
      snprintf(id_str, 4, "%03d", frame_id + 1);
      cv::Mat frame;

      try {
        cv::cvtColor(cv::imread(frames_path + std::string(id_str) + ".png"), frame, cv::COLOR_BGR2RGB);
      } catch (...) { break; }

      ++frame_id;
      frames.push_back(frame);
    }

    out_frames.push_back(frames);
  }
}

void VideoTest::ComapreFrames(const uint8_t *frame, const uint8_t *gt, int size, int eps) {
    for (int j = 0; j < size; ++j) {
      ASSERT_NEAR(frame[j], gt[j], eps);
    }
}
}  // namespace dali