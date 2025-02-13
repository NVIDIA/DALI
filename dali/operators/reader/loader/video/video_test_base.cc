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

#include "dali/operators/reader/loader/video/video_test_base.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <thread>

#include "dali/test/dali_test_config.h"
#include "dali/test/cv_mat_utils.h"

namespace dali {
namespace detail {
static void parallel_for(
  int elements_count, int thread_count, std::function<void(int start, int end, int id)> func) {
  std::vector<std::thread> threads(thread_count);

  for (int i = 0; i < thread_count; ++i) {
    int start = elements_count * i / thread_count;
    int end = elements_count * (i+1) / thread_count;
    threads[i] = std::thread(func, start, end, i);
  }

  std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
}

int ThreadCount() {
  int num_thread_hint = std::thread::hardware_concurrency();
  return num_thread_hint == 0 ? 8 : num_thread_hint;
}
}  // namespace detail

void CompareFrameAvgError(
  const uint8_t *ground_truth, const uint8_t *frame, int frame_size, double eps) {
  std::vector<double> sums(detail::ThreadCount(), 0.0);

  detail::parallel_for(frame_size, detail::ThreadCount(), [&](int start, int end, int id){
    double sum = 0.0;
    for (int j = start; j < end; ++j) {
      sum += std::abs(frame[j] - ground_truth[j]);
    }
    sums[id] = sum;
  });

  double sum = std::accumulate(sums.begin(), sums.end(), 0.0);
  sum /= frame_size;

  ASSERT_LT(sum, eps);
}

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
void SaveFrame(uint8_t *frame, int frame_id, int sample_id, int batch_id,
              const std::string &prefix, int width, int height) {
  TensorView<StorageCPU, uint8_t> tv(frame, TensorShape<3>{height, width, 3});
  char str[32];
  snprintf(str, sizeof(str), "_%03d_%03d_%03d.png", batch_id, sample_id, frame_id);
  string full_path = prefix + string(str);
  std::cout << "Writing " << full_path << std::endl;
  testing::SaveImage(full_path.c_str(), tv);
}

void TestVideo::CompareFrame(int frame_id, const uint8_t *frame, int eps) {
  auto &ground_truth = frames_[frame_id];
  bool frames_match = true;

  detail::parallel_for(FrameSize(), detail::ThreadCount(), [&](int start, int end, int id){
    for (int j = start; j < end; ++j) {
      if (std::abs(frame[j] - ground_truth.data[j]) > eps) {
        frames_match = false;
        break;
      }
    }
  });

  if (!frames_match) {
    SaveFrame(const_cast<uint8_t *>(frame), frame_id, 0, 0, "test_frame", Width(),
              Height());
    SaveFrame(ground_truth.data, frame_id, 0, 0, "ground_truth", Width(),
              Height());

    FAIL() << "Frames do not match (eps=" << eps
           << "). Debug frames saved to test_frame_*.png and ground_truth_*.png";
  }
}

void TestVideo::CompareFrameAvgError(int frame_id, const uint8_t *frame, double eps) {
  auto &ground_truth = frames_[frame_id];

  dali::CompareFrameAvgError(ground_truth.data, frame, FrameSize(), eps);
}

std::vector<std::string> VideoTestBase::cfr_videos_frames_paths_{
  testing::dali_extra_path() + "/db/video/cfr/frames_1/",
  testing::dali_extra_path() + "/db/video/cfr/frames_2/"};

std::vector<std::string> VideoTestBase::vfr_videos_frames_paths_{
  testing::dali_extra_path() + "/db/video/vfr/frames_1/",
  testing::dali_extra_path() + "/db/video/vfr/frames_2/"};

std::vector<std::string> VideoTestBase::vfr_hevc_videos_frames_paths_{
  testing::dali_extra_path() + "/db/video/vfr/frames_1_hevc/",
  testing::dali_extra_path() + "/db/video/vfr/frames_2_hevc/"};

std::vector<std::string> VideoTestBase::cfr_videos_paths_{
  testing::dali_extra_path() + "/db/video/cfr/test_1.mp4",
  testing::dali_extra_path() + "/db/video/cfr/test_2.mp4"};

std::vector<std::string> VideoTestBase::vfr_videos_paths_{
  testing::dali_extra_path() + "/db/video/vfr/test_1.mp4",
  testing::dali_extra_path() + "/db/video/vfr/test_2.mp4"};

std::vector<std::string> VideoTestBase::cfr_hevc_videos_paths_{
  testing::dali_extra_path() + "/db/video/cfr/test_1_hevc.mp4",
  testing::dali_extra_path() + "/db/video/cfr/test_2_hevc.mp4"};

std::vector<std::string> VideoTestBase::vfr_hevc_videos_paths_{
  testing::dali_extra_path() + "/db/video/vfr/test_1_hevc.mp4",
  testing::dali_extra_path() + "/db/video/vfr/test_2_hevc.mp4"};

std::vector<std::string> VideoTestBase::cfr_mpeg4_videos_paths_{
  testing::dali_extra_path() + "/db/video/cfr/test_1_mpeg4.mp4",
  testing::dali_extra_path() + "/db/video/cfr/test_2_mpeg4.mp4"};

std::vector<std::string> VideoTestBase::vfr_mpeg4_videos_paths_{
  testing::dali_extra_path() + "/db/video/vfr/test_1_mpeg4.mp4",
  testing::dali_extra_path() + "/db/video/vfr/test_2_mpeg4.mp4"};

std::vector<std::string> VideoTestBase::cfr_mpeg4_mkv_videos_paths_{
  testing::dali_extra_path() + "/db/video/cfr/test_1_mpeg4.mkv",
  testing::dali_extra_path() + "/db/video/cfr/test_2_mpeg4.mkv"};

std::vector<std::string> VideoTestBase::vfr_mpeg4_mkv_videos_paths_{
  testing::dali_extra_path() + "/db/video/vfr/test_1_mpeg4.mkv",
  testing::dali_extra_path() + "/db/video/vfr/test_2_mpeg4.mkv"};

std::vector<std::string> VideoTestBase::cfr_raw_h264_videos_paths_{
  testing::dali_extra_path() + "/db/video/cfr/test_1.h264",
  testing::dali_extra_path() + "/db/video/cfr/test_2.h264"};

std::vector<std::string> VideoTestBase::cfr_raw_h265_videos_paths_{
  testing::dali_extra_path() + "/db/video/cfr/test_1.h265",
  testing::dali_extra_path() + "/db/video/cfr/test_2.h265"};

std::vector<TestVideo> VideoTestBase::cfr_videos_;
std::vector<TestVideo> VideoTestBase::vfr_videos_;
std::vector<TestVideo> VideoTestBase::vfr_hevc_videos_;

void VideoTestBase::SetUpTestSuite() {
  if (cfr_videos_.size() > 0) {
    // This setup is called for each test fixture, but we need it only once
    return;
  }

  for (auto &folder_path : cfr_videos_frames_paths_) {
    cfr_videos_.push_back(TestVideo(folder_path, false));
  }

  for (auto &folder_path : vfr_videos_frames_paths_) {
    vfr_videos_.push_back(TestVideo(folder_path, true));
  }

  for (auto &folder_path : vfr_hevc_videos_frames_paths_) {
    vfr_hevc_videos_.push_back(TestVideo(folder_path, true));
  }
}

void VideoTestBase::RunFailureTest(std::function<void()> body, std::string expected_error) {
  try {
    body();

    FAIL();   // If we reached this point body did not throw an exception.
  } catch (const std::exception &e) {
    EXPECT_TRUE(
      strstr(e.what(), expected_error.c_str()));
  }
}

std::vector<char> VideoTestBase::MemoryVideo(const std::string &path) const {
  std::ifstream video_file(path, std::ios::binary | std::ios::ate);
  auto size = video_file.tellg();
  video_file.seekg(0, std::ios::beg);

  std::vector<char> memory_video(size);
  if (!video_file.read(memory_video.data(), size)) {
    // We can't use FAIL() because this function returns value
    throw ::testing::AssertionFailure() << "Could not load video file to memory.";
  }

  return memory_video;
}

}  // namespace dali
