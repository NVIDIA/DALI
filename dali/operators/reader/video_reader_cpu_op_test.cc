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

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "dali/pipeline/pipeline.h"
#include "dali/test/dali_test_config.h"


namespace dali {

namespace detail {
/// @param[in] nb_elements : size of your for loop
/// @param[in] functor(start, end) :
/// your function processing a sub chunk of the for loop.
/// "start" is the first index to process (included) until the index "end"
/// (excluded)
/// @code
///     for(int i = start; i < end; ++i)
///         computation(i);
/// @endcode
/// @param use_threads : enable / disable threads.
///
///
static void parallel_for(unsigned nb_elements,
                  std::function<void (int start, int end)> functor,
                  bool use_threads = true)
{
    // -------
    unsigned nb_threads_hint = std::thread::hardware_concurrency();
    unsigned nb_threads = nb_threads_hint == 0 ? 8 : (nb_threads_hint);

    unsigned batch_size = nb_elements / nb_threads;
    unsigned batch_remainder = nb_elements % nb_threads;

    std::vector< std::thread > my_threads(nb_threads);

    if( use_threads )
    {
        // Multithread execution
        for(unsigned i = 0; i < nb_threads; ++i)
        {
            int start = i * batch_size;
            my_threads[i] = std::thread(functor, start, start+batch_size);
        }
    }
    else
    {
        // Single thread execution (for easy debugging)
        for(unsigned i = 0; i < nb_threads; ++i){
            int start = i * batch_size;
            functor( start, start+batch_size );
        }
    }

    // Deform the elements left
    int start = nb_threads * batch_size;
    functor( start, start+batch_remainder);

    // Wait for the other thread to finish their task
    if( use_threads )
        std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
}

void comapre_frames(const uint8_t *frame, const uint8_t *gt, size_t size) {
  detail::parallel_for(size, [&](int start, int end){ 
    for (int j = start; j < end; ++j) {
      ASSERT_EQ(frame[j], gt[j]);
    }});
}

} // namespace detail

class VideoReaderCpuTest : public ::testing::Test {
 public:
  VideoReaderCpuTest() {
    std::string frames_path = testing::dali_extra_path() + "/db/video/cfr/frames/";
    char id_str[4];
  
    for (int i = 0; i < NumFrames(); ++i) {
      snprintf(id_str, 4, "%03d", i + 1);
      cv::Mat frame;
      cv::cvtColor(
        cv::imread(frames_path + string(id_str) + ".png"),
        frame,
        cv::COLOR_BGR2RGB);
      gt_frames_.push_back(frame);
    }
  }

  const int NumFrames() const { return 50; }

  const int Channels() const { return 3; }

  const int Width() const { return gt_frames_[0].cols; }

  const int Height() const { return gt_frames_[0].rows; }

  const int FrameSize() const { return Height() * Width() * Channels(); }

 protected:
  std::vector<cv::Mat> gt_frames_;
};


TEST_F(VideoReaderCpuTest, CpuConstantFrameRate) {
  Pipeline pipe(1, 4, 0);
  const int sequence_length = 1;

  pipe.AddOperator(OpSpec("readers__Video")
    .AddArg("device", "cpu")
    .AddArg("sequence_length", sequence_length)
    .AddArg(
      "filenames",
      std::vector<std::string>{
        testing::dali_extra_path() + "/db/video/cfr/test.mp4"})
    .AddOutput("frames", "cpu"));

  pipe.Build({{"frames", "cpu"}});

  for (int frame_id = 0; frame_id < this->NumFrames() * 2 + 10; ++frame_id) {
    DeviceWorkspace ws;
    pipe.RunCPU();
    pipe.RunGPU();
    pipe.Outputs(&ws);

    const auto &frame_video_output = ws.Output<dali::CPUBackend>(0);

    detail::comapre_frames(
      frame_video_output.data<uint8_t>(), this->gt_frames_[frame_id % this->NumFrames()].data, this->FrameSize());
  }
}
}  // namespace dali
