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

#include <cuda_runtime_api.h>
#include <exception>
#include <random>

#include "dali/core/cuda_error.h"
#include "dali/core/dev_buffer.h"
#include "dali/core/device_guard.h"
#include "dali/core/dynlink_cuda.h"
#include "dali/core/error_handling.h"
#include "dali/operators/reader/loader/video/frames_decoder.h"
#include "dali/operators/reader/loader/video/frames_decoder_gpu.h"
#include "dali/operators/reader/loader/video/video_test_base.h"
#include "dali/test/dali_test_config.h"

#include "dali/pipeline/pipeline.h"

namespace dali {
class FramesDecoderTestBase : public VideoTestBase {};

class FramesDecoderTest_CpuOnlyTests : public FramesDecoderTestBase {
 public:
  void RunTest(std::string file_name, TestVideo &ground_truth) {
    FramesDecoder file(testing::dali_extra_path() + file_name);

    ASSERT_EQ(file.Height(), ground_truth.Height());
    ASSERT_EQ(file.Width(), ground_truth.Width());
    ASSERT_EQ(file.Channels(), ground_truth.NumChannels());
    ASSERT_EQ(file.NumFrames(), ground_truth.NumFrames());

    std::vector<uint8_t> frame(file.FrameSize());

    // Iterate through the whole video in order
    for (int i = 0; i < file.NumFrames(); ++i) {
      ASSERT_EQ(file.NextFrameIdx(), i);
      file.ReadNextFrame(frame.data());
      ground_truth.CompareFrame(i, frame.data());
    }

    ASSERT_EQ(file.NextFrameIdx(), -1);
    file.Reset();

    // Read first frame
    ASSERT_EQ(file.NextFrameIdx(), 0);
    file.ReadNextFrame(frame.data());
    ground_truth.CompareFrame(0, frame.data());

    // Seek to frame
    file.SeekFrame(25);
    ASSERT_EQ(file.NextFrameIdx(), 25);
    file.ReadNextFrame(frame.data());
    ground_truth.CompareFrame(25, frame.data());

    // Seek back to frame
    file.SeekFrame(12);
    ASSERT_EQ(file.NextFrameIdx(), 12);
    file.ReadNextFrame(frame.data());
    ground_truth.CompareFrame(12, frame.data());

    // Seek to last frame (flush frame)
    int last_frame_index = ground_truth.NumFrames() - 1;
    file.SeekFrame(last_frame_index);
    ASSERT_EQ(file.NextFrameIdx(), last_frame_index);
    file.ReadNextFrame(frame.data());
    ground_truth.CompareFrame(last_frame_index, frame.data());
    ASSERT_EQ(file.NextFrameIdx(), -1);

    // Wrap around to first frame
    ASSERT_FALSE(file.ReadNextFrame(frame.data()));
    file.Reset();
    ASSERT_EQ(file.NextFrameIdx(), 0);
    file.ReadNextFrame(frame.data());
    ground_truth.CompareFrame(0, frame.data());

    // Seek to random frames and read them
    std::mt19937 gen(0);
    std::uniform_int_distribution<> distr(0, last_frame_index);

    for (int i = 0; i < 20; ++i) {
      int next_index = distr(gen);

      file.SeekFrame(next_index);
      file.ReadNextFrame(frame.data());
      ground_truth.CompareFrame(next_index, frame.data());
    }
  }
};

class FramesDecoderGpuTest : public FramesDecoderTestBase {
 public:
  static void SetUpTestSuite() {
    VideoTestBase::SetUpTestSuite();
    DeviceGuard(0);
    CUDA_CALL(cudaDeviceSynchronize());
  }

  void RunTest(std::string file_name, TestVideo &ground_truth) {
    FramesDecoderGpu file(testing::dali_extra_path() + file_name);

    ASSERT_EQ(file.Height(), ground_truth.Height());
    ASSERT_EQ(file.Width(), ground_truth.Width());
    ASSERT_EQ(file.Channels(), ground_truth.NumChannels());
    ASSERT_EQ(file.NumFrames(), ground_truth.NumFrames());

    DeviceBuffer<uint8_t> frame;
    frame.resize(file.FrameSize());
    std::vector<uint8_t> frame_cpu(file.FrameSize());

    // Iterate through the whole video in order
    for (int i = 0; i < file.NumFrames(); ++i) {
      ASSERT_EQ(file.NextFrameIdx(), i);
      file.ReadNextFrame(frame.data());
      MemCopy(frame_cpu.data(), frame.data(), file.FrameSize());
      ground_truth.CompareFrameAvgError(i, frame_cpu.data());
    }

    ASSERT_EQ(file.NextFrameIdx(), -1);
    file.Reset();

    // Read first frame
    ASSERT_EQ(file.NextFrameIdx(), 0);
    file.ReadNextFrame(frame.data());
    MemCopy(frame_cpu.data(), frame.data(), file.FrameSize());
    ground_truth.CompareFrameAvgError(0, frame_cpu.data());

    // Seek to frame
    file.SeekFrame(25);
    ASSERT_EQ(file.NextFrameIdx(), 25);
    file.ReadNextFrame(frame.data());
    MemCopy(frame_cpu.data(), frame.data(), file.FrameSize());
    ground_truth.CompareFrameAvgError(25, frame_cpu.data());

    // Seek back to frame
    file.SeekFrame(12);
    ASSERT_EQ(file.NextFrameIdx(), 12);
    file.ReadNextFrame(frame.data());
    MemCopy(frame_cpu.data(), frame.data(), file.FrameSize());
    ground_truth.CompareFrameAvgError(12, frame_cpu.data());

    // Seek to last frame (flush frame)
    int last_frame_index = ground_truth.NumFrames() - 1;
    file.SeekFrame(last_frame_index);
    ASSERT_EQ(file.NextFrameIdx(), last_frame_index);
    file.ReadNextFrame(frame.data());
    MemCopy(frame_cpu.data(), frame.data(), file.FrameSize());
    ground_truth.CompareFrameAvgError(last_frame_index, frame_cpu.data());
    ASSERT_EQ(file.NextFrameIdx(), -1);

    // Wrap around to first frame
    ASSERT_FALSE(file.ReadNextFrame(frame.data()));
    file.Reset();
    ASSERT_EQ(file.NextFrameIdx(), 0);
    file.ReadNextFrame(frame.data());
    MemCopy(frame_cpu.data(), frame.data(), file.FrameSize());
    ground_truth.CompareFrameAvgError(0, frame_cpu.data());

    // Seek to random frames and read them
    std::mt19937 gen(0);
    std::uniform_int_distribution<> distr(0, last_frame_index);

    for (int i = 0; i < 20; ++i) {
      int next_index = distr(gen);

      file.SeekFrame(next_index);
      file.ReadNextFrame(frame.data());
      MemCopy(frame_cpu.data(), frame.data(), file.FrameSize());
      ground_truth.CompareFrameAvgError(next_index, frame_cpu.data());
    }
  }
};


TEST_F(FramesDecoderTest_CpuOnlyTests, ConstantFrameRate) {
  RunTest("/db/video/cfr/test_1.mp4", cfr_videos_[0]);
}

TEST_F(FramesDecoderTest_CpuOnlyTests, VariableFrameRate) {
  RunTest("/db/video/vfr/test_2.mp4", vfr_videos_[1]);
}

TEST_F(FramesDecoderTest_CpuOnlyTests, InvalidPath) {
  std::string path = "invalid_path.mp4";

  try {
    FramesDecoder file(path);
  } catch (const DALIException &e) {
    EXPECT_TRUE(strstr(e.what(), make_string("Failed to open video file at path ", path).c_str()));
  }
}

TEST_F(FramesDecoderTest_CpuOnlyTests, NoVideoStream) {
  std::string path = testing::dali_extra_path() + "/db/audio/wav/dziendobry.wav";

  try {
    FramesDecoder file(path);
  } catch (const DALIException &e) {
    EXPECT_TRUE(strstr(
        e.what(), make_string("Could not find a valid video stream in a file ", path).c_str()));
  }
}

TEST_F(FramesDecoderTest_CpuOnlyTests, InvalidSeek) {
  std::string path = testing::dali_extra_path() + "/db/video/cfr/test_1.mp4";
  FramesDecoder file(path);

  try {
    file.SeekFrame(60);
  } catch (const DALIException &e) {
    EXPECT_TRUE(strstr(e.what(), "Invalid seek frame id. frame_id = 60, num_frames = 50"));
  }
}

TEST_F(FramesDecoderGpuTest, ConstantFrameRate) {
  RunTest("/db/video/cfr/test_1.mp4", cfr_videos_[0]);
}

TEST_F(FramesDecoderGpuTest, VariableFrameRate) {
  RunTest("/db/video/vfr/test_2.mp4", vfr_videos_[1]);
}

// TEST_F(FramesDecoderTest_CpuOnlyTests, Main) {
//     DeviceGuard(0);
//     CUDA_CALL(cudaDeviceSynchronize());

//     std::string path = "/home/awolant/Downloads/test_1080.mp4";
// //     // std::string path = "/home/awolant/Downloads/test_1080_gop.mp4";
// //     // std::string path = "/home/awolant/Downloads/test_720.mp4";

//     FramesDecoderGpu file(path);
//     FramesDecoder file_cpu(path);

//     DeviceBuffer<uint8_t> frame;
//     std::vector<uint8_t> frame_cpu;
//     frame.resize(file.FrameSize());
//     frame_cpu.resize(file.FrameSize());

//     for (int i = 0; i < 100; ++i) {
//         file.SeekFrame(i);
//         file.ReadNextFrame(frame);
//         MemCopy(frame_cpu.data(), frame.data(), file.FrameSize());
//         SaveFrame(frame_cpu.data(), i, 0, 0, "/home/awolant/Downloads/frames/decoder_gpu/",
//         file.Width(), file.Height());

//         file_cpu.SeekFrame(i);
//         file_cpu.ReadNextFrame(frame_cpu.data());
//         SaveFrame(frame_cpu.data(), i, 0, 0, "/home/awolant/Downloads/frames/decoder_cpu/",
//         file_cpu.Width(), file_cpu.Height());

//     }


// //     // const int batch_size = 1;
// //     // const int sequence_length = 1;
// //     // const int stride = 1;
// //     // const int step = 1;
// //     // const int shard_id = 0;
// //     // const int num_shards = 1;
// //     // const int seed = 0;
// //     // const int initial_fill = 0;

// //     // Pipeline pipe(batch_size, 1, 0);

// //     // pipe.AddOperator(OpSpec("readers__Video")
// //     //     .AddArg("device", "gpu")
// //     //     .AddArg("sequence_length", sequence_length)
// //     //     .AddArg("stride", stride)
// //     //     .AddArg("step", step)
// //     //     .AddArg("shard_id ", shard_id)
// //     //     .AddArg("num_shards ", num_shards)
// //     //     .AddArg("seed", seed)
// //     //     .AddArg("initial_fill", initial_fill)
// //     //     .AddArg("random_shuffle", false)
// //     //     .AddArg(
// //     //         "filenames",
// //     //         std::vector<std::string>{path})
// //     //     .AddArg("labels", std::vector<int>{0})
// //     //     .AddOutput("frames_gpu", "gpu")
// //     //     .AddOutput("labels_gpu", "gpu"));

// //     // pipe.Build({{"frames_gpu", "gpu"}, {"labels_gpu", "gpu"}});

// //     // DeviceWorkspace ws;
// //     // for (int i = 0; i < 81; ++i) {
// //     //     pipe.RunCPU();
// //     //     pipe.RunGPU();
// //     //     pipe.Outputs(&ws);
// //     // }

// //     // std::cout <<
// "\n\n=======================================================================\n\n";

// //     // pipe.RunCPU();
// //     // pipe.RunGPU();
// //     // pipe.Outputs(&ws);

// //     // for (int i = 0; i < 81; ++i) {
// //     //     pipe.RunCPU();
// //     //     pipe.RunGPU();
// //     //     pipe.Outputs(&ws);
// //     // }
// }

}  // namespace dali
