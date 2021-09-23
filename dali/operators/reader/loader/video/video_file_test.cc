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

#include <exception>

#include "dali/test/dali_test_config.h"
#include "dali/core/error_handling.h"
#include "dali/operators/reader/loader/video/video_test_base.h"
#include "dali/operators/reader/loader/video/video_file.h"


namespace dali {
class VideoFileTest : public VideoTestBase {
};


TEST_F(VideoFileTest, ConstantFrameRate) {
    std::string path = testing::dali_extra_path() + "/db/video/cfr/test_1.mp4"; 
    
    // Create file, build index
    VideoFileCPU file(path);

    ASSERT_EQ(file.Height(), 720);
    ASSERT_EQ(file.Width(), 1280);
    ASSERT_EQ(file.Channels(), 3);
    ASSERT_EQ(file.NumFrames(), 50);

    std::vector<uint8_t> frame(file.FrameSize());

    // Read first frame
    file.ReadNextFrame(frame.data());
    this->ComapreFrames(frame.data(), this->GetCfrFrame(0, 0), file.FrameSize());

    // Seek to frame
    file.SeekFrame(25);
    file.ReadNextFrame(frame.data());
    this->ComapreFrames(frame.data(), this->GetCfrFrame(0, 25), file.FrameSize());

    // Seek back to frame
    file.SeekFrame(12);
    file.ReadNextFrame(frame.data());
    this->ComapreFrames(frame.data(), this->GetCfrFrame(0, 12), file.FrameSize());

    // Seek to last frame (flush frame)
    file.SeekFrame(49);
    file.ReadNextFrame(frame.data());
    this->ComapreFrames(frame.data(), this->GetCfrFrame(0, 49), file.FrameSize());

    // Wrap around to first frame
    ASSERT_FALSE(file.ReadNextFrame(frame.data()));
    file.Reset();
    file.ReadNextFrame(frame.data());
    this->ComapreFrames(frame.data(), this->GetCfrFrame(0, 0), file.FrameSize());
}

TEST_F(VideoFileTest, VariableFrameRate) {
    std::string path = testing::dali_extra_path() + "/db/video/vfr/test_2.mp4"; 
    
    // Create file, build index
    VideoFileCPU file(path);

    ASSERT_EQ(file.Height(), 600);
    ASSERT_EQ(file.Width(), 800);
    ASSERT_EQ(file.Channels(), 3);
    ASSERT_EQ(file.NumFrames(), 60);

    std::vector<uint8_t> frame(file.FrameSize());

    // Read first frame
    file.ReadNextFrame(frame.data());
    this->ComapreFrames(frame.data(), this->GetVfrFrame(1,0), file.FrameSize());

    // Seek to frame
    file.SeekFrame(25);
    file.ReadNextFrame(frame.data());
    this->ComapreFrames(frame.data(), this->GetVfrFrame(1, 25), file.FrameSize());

    // Seek back to frame
    file.SeekFrame(12);
    file.ReadNextFrame(frame.data());
    this->ComapreFrames(frame.data(), this->GetVfrFrame(1, 12), file.FrameSize());

    // Seek to last frame (flush frame)
    file.SeekFrame(59);
    file.ReadNextFrame(frame.data());
    this->ComapreFrames(frame.data(), this->GetVfrFrame(1, 59), file.FrameSize());

    // Wrap around to first frame
    ASSERT_FALSE(file.ReadNextFrame(frame.data()));
    file.Reset();
    file.ReadNextFrame(frame.data());
    this->ComapreFrames(frame.data(), this->GetVfrFrame(1, 0), file.FrameSize());
}

TEST_F(VideoFileTest, InvalidPath) {
    std::string path = "invalid_path.mp4"; 
    
    try {
        VideoFileCPU file(path);
    } catch (const DALIException &e) {
        EXPECT_TRUE(strstr(
            e.what(),
            make_string("Failed to open video file at path ", path).c_str()));
    }
}

TEST_F(VideoFileTest, NoVideoStream) {
    std::string path = testing::dali_extra_path() + "/db/audio/wav/dziendobry.wav"; 
    
    try {
        VideoFileCPU file(path);
    } catch (const DALIException &e) {
        EXPECT_TRUE(strstr(
            e.what(),
            make_string("Could not find a valid video stream in a file ", path).c_str()));
    }
}

TEST_F(VideoFileTest, InvalidSeek) {
    std::string path = testing::dali_extra_path() + "/db/video/cfr/test_1.mp4";
    VideoFileCPU file(path);

    try {
        file.SeekFrame(60);
    } catch (const DALIException &e) {
        EXPECT_TRUE(strstr(
            e.what(),
            "Invalid seek frame id. frame_id = 60, num_frames = 50"));
    }
}

}  // namespace dali
