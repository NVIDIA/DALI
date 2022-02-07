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

#include "dali/test/dali_test_config.h"
#include "dali/core/error_handling.h"
#include "dali/core/dynlink_cuda.h"
#include "dali/core/cuda_error.h"
#include "dali/core/dev_buffer.h"
#include "dali/core/device_guard.h"
#include "dali/operators/reader/loader/video/video_test_base.h"
#include "dali/operators/reader/loader/video/frames_decoder.h"
#include "dali/operators/reader/loader/video/frames_decoder_gpu.h"


namespace dali {
class FramesDecoderTest : public VideoTestBase {
};


TEST_F(FramesDecoderTest, ConstantFrameRate) {
    std::string path = testing::dali_extra_path() + "/db/video/cfr/test_1.mp4";

    // Create file, build index
    FramesDecoder file(path);

    ASSERT_EQ(file.Height(), 720);
    ASSERT_EQ(file.Width(), 1280);
    ASSERT_EQ(file.Channels(), 3);
    ASSERT_EQ(file.NumFrames(), 50);

    std::vector<uint8_t> frame(file.FrameSize());

    // Read first frame
    ASSERT_EQ(file.NextFrameIdx(), 0);
    file.ReadNextFrame(frame.data());
    this->CompareFrames(frame.data(), this->GetCfrFrame(0, 0), file.FrameSize());

    // Seek to frame
    file.SeekFrame(25);
    ASSERT_EQ(file.NextFrameIdx(), 25);
    file.ReadNextFrame(frame.data());
    this->CompareFrames(frame.data(), this->GetCfrFrame(0, 25), file.FrameSize());

    // Seek back to frame
    file.SeekFrame(12);
    ASSERT_EQ(file.NextFrameIdx(), 12);
    file.ReadNextFrame(frame.data());
    this->CompareFrames(frame.data(), this->GetCfrFrame(0, 12), file.FrameSize());

    // Seek to last frame (flush frame)
    file.SeekFrame(49);
    ASSERT_EQ(file.NextFrameIdx(), 49);
    file.ReadNextFrame(frame.data());
    this->CompareFrames(frame.data(), this->GetCfrFrame(0, 49), file.FrameSize());
    ASSERT_EQ(file.NextFrameIdx(), -1);

    // Wrap around to first frame
    ASSERT_FALSE(file.ReadNextFrame(frame.data()));
    file.Reset();
    ASSERT_EQ(file.NextFrameIdx(), 0);
    file.ReadNextFrame(frame.data());
    this->CompareFrames(frame.data(), this->GetCfrFrame(0, 0), file.FrameSize());
}

TEST_F(FramesDecoderTest, VariableFrameRateCpu) {
    std::string path = testing::dali_extra_path() + "/db/video/vfr/test_2.mp4";

    // Create file, build index
    FramesDecoder file(path);

    ASSERT_EQ(file.Height(), 600);
    ASSERT_EQ(file.Width(), 800);
    ASSERT_EQ(file.Channels(), 3);
    ASSERT_EQ(file.NumFrames(), 60);

    std::vector<uint8_t> frame(file.FrameSize());

    // Read first frame
    ASSERT_EQ(file.NextFrameIdx(), 0);
    file.ReadNextFrame(frame.data());
    this->CompareFrames(frame.data(), this->GetVfrFrame(1, 0), file.FrameSize());

    // Seek to frame
    file.SeekFrame(25);
    ASSERT_EQ(file.NextFrameIdx(), 25);
    file.ReadNextFrame(frame.data());
    this->CompareFrames(frame.data(), this->GetVfrFrame(1, 25), file.FrameSize());

    // Seek back to frame
    file.SeekFrame(12);
    ASSERT_EQ(file.NextFrameIdx(), 12);
    file.ReadNextFrame(frame.data());
    this->CompareFrames(frame.data(), this->GetVfrFrame(1, 12), file.FrameSize());

    // Seek to last frame (flush frame)
    file.SeekFrame(59);
    ASSERT_EQ(file.NextFrameIdx(), 59);
    file.ReadNextFrame(frame.data());
    this->CompareFrames(frame.data(), this->GetVfrFrame(1, 59), file.FrameSize());

    // Wrap around to first frame
    ASSERT_FALSE(file.ReadNextFrame(frame.data()));
    file.Reset();
    ASSERT_EQ(file.NextFrameIdx(), 0);
    file.ReadNextFrame(frame.data());
    this->CompareFrames(frame.data(), this->GetVfrFrame(1, 0), file.FrameSize());
}

TEST_F(FramesDecoderTest, VariableFrameRateGpu) {
    DeviceGuard(0);
    CUDA_CALL(cudaDeviceSynchronize());

    std::string path = testing::dali_extra_path() + "/db/video/vfr/test_2.mp4";

    // Create file, build index
    FramesDecoderGpu file(path);

    ASSERT_EQ(file.Height(), 600);
    ASSERT_EQ(file.Width(), 800);
    ASSERT_EQ(file.Channels(), 3);
    ASSERT_EQ(file.NumFrames(), 60);

    DeviceBuffer<uint8_t> frame;
    frame.resize(file.FrameSize());
    std::vector<uint8_t> frame_cpu(file.FrameSize());

    for (int i = 0; i < file.NumFrames(); ++i) {
        ASSERT_EQ(file.NextFrameIdx(), i);
        file.ReadNextFrame(frame.data());
        copyD2H(frame_cpu.data(), frame.data(), frame.size());
        this->CompareFramesAvgError(frame_cpu.data(), this->GetVfrFrame(1, i), file.FrameSize());
    }
    ASSERT_EQ(file.NextFrameIdx(), -1);
    file.Reset();
    ASSERT_EQ(file.NextFrameIdx(), 0);

    // Read first frame
    file.ReadNextFrame(frame);
    copyD2H(frame_cpu.data(), frame.data(), frame.size());
    this->CompareFramesAvgError(frame_cpu.data(), this->GetVfrFrame(1, 0), file.FrameSize());

    // Seek to frame
    file.SeekFrame(25);
    file.ReadNextFrame(frame);
    copyD2H(frame_cpu.data(), frame.data(), frame.size());
    this->CompareFramesAvgError(frame_cpu.data(), this->GetVfrFrame(1, 25), file.FrameSize());

    // Seek back to frame
    file.SeekFrame(12);
    file.ReadNextFrame(frame);
    copyD2H(frame_cpu.data(), frame.data(), frame.size());
    this->CompareFramesAvgError(frame_cpu.data(), this->GetVfrFrame(1, 12), file.FrameSize());

    // Seek to last frame (flush frame)
    file.SeekFrame(59);
    file.ReadNextFrame(frame);
    copyD2H(frame_cpu.data(), frame.data(), frame.size());
    this->CompareFramesAvgError(frame_cpu.data(), this->GetVfrFrame(1, 59), file.FrameSize(), 1.1);
    ASSERT_EQ(file.NextFrameIdx(), -1);

    // Wrap around to first frame
    ASSERT_FALSE(file.ReadNextFrame(frame));
    file.Reset();
    file.ReadNextFrame(frame);
    copyD2H(frame_cpu.data(), frame.data(), frame.size());
    this->CompareFramesAvgError(frame_cpu.data(), this->GetVfrFrame(1, 0), file.FrameSize());

    // Seek to random frames and read them
    std::mt19937 gen(0);
    std::uniform_int_distribution<> distr(0, file.NumFrames() - 1);

    for (int i = 0; i < 20; ++i) {
        int next_index = distr(gen);

        file.SeekFrame(next_index);
        file.ReadNextFrame(frame);
        copyD2H(frame_cpu.data(), frame.data(), frame.size());
        this->CompareFramesAvgError(
            frame_cpu.data(), this->GetVfrFrame(1, next_index), file.FrameSize());
    }
}


TEST_F(FramesDecoderTest, InvalidPath) {
    std::string path = "invalid_path.mp4";

    try {
        FramesDecoder file(path);
    } catch (const DALIException &e) {
        EXPECT_TRUE(strstr(
            e.what(),
            make_string("Failed to open video file at path ", path).c_str()));
    }
}

TEST_F(FramesDecoderTest, NoVideoStream) {
    std::string path = testing::dali_extra_path() + "/db/audio/wav/dziendobry.wav";

    try {
        FramesDecoder file(path);
    } catch (const DALIException &e) {
        EXPECT_TRUE(strstr(
            e.what(),
            make_string("Could not find a valid video stream in a file ", path).c_str()));
    }
}

TEST_F(FramesDecoderTest, InvalidSeek) {
    std::string path = testing::dali_extra_path() + "/db/video/cfr/test_1.mp4";
    FramesDecoder file(path);

    try {
        file.SeekFrame(60);
    } catch (const DALIException &e) {
        EXPECT_TRUE(strstr(
            e.what(),
            "Invalid seek frame id. frame_id = 60, num_frames = 50"));
    }
}

}  // namespace dali
