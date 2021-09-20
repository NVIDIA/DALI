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

#include "dali/test/dali_test_config.h"
#include "dali/operators/reader/loader/video/video_test.h"
#include "dali/operators/reader/loader/video/video_file.h"


namespace dali {
class VideoFileTest : public VideoTest {
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
    this->ComapreFrames(frame.data(), this->cfr_frames_[0][0].data, file.FrameSize());

    // Seek to frame
    file.SeekFrame(25);
    file.ReadNextFrame(frame.data());
    this->ComapreFrames(frame.data(), this->cfr_frames_[0][25].data, file.FrameSize());

    // Seek back to frame
    file.SeekFrame(12);
    file.ReadNextFrame(frame.data());
    this->ComapreFrames(frame.data(), this->cfr_frames_[0][12].data, file.FrameSize());

    // Seek to last frame (flush frame)
    file.SeekFrame(49);
    file.ReadNextFrame(frame.data());
    this->ComapreFrames(frame.data(), this->cfr_frames_[0][49].data, file.FrameSize());

    // Wrap around to first frame
    file.ReadNextFrame(frame.data());
    this->ComapreFrames(frame.data(), this->cfr_frames_[0][0].data, file.FrameSize());
}

TEST_F(VideoFileTest, VariableFrameRate) {
    std::string path = testing::dali_extra_path() + "/db/video/vfr/test_1.mp4"; 
    
    // Create file, build index
    VideoFileCPU file(path);

    ASSERT_EQ(file.Height(), 720);
    ASSERT_EQ(file.Width(), 1280);
    ASSERT_EQ(file.Channels(), 3);
    ASSERT_EQ(file.NumFrames(), 50);

    std::vector<uint8_t> frame(file.FrameSize());

    // Read first frame
    file.ReadNextFrame(frame.data());
    this->ComapreFrames(frame.data(), this->vfr_frames_[0][0].data, file.FrameSize());

    // Seek to frame
    file.SeekFrame(25);
    file.ReadNextFrame(frame.data());
    this->ComapreFrames(frame.data(), this->vfr_frames_[0][25].data, file.FrameSize());

    // Seek back to frame
    file.SeekFrame(12);
    file.ReadNextFrame(frame.data());
    this->ComapreFrames(frame.data(), this->vfr_frames_[0][12].data, file.FrameSize());

    // Seek to last frame (flush frame)
    file.SeekFrame(49);
    file.ReadNextFrame(frame.data());
    this->ComapreFrames(frame.data(), this->vfr_frames_[0][49].data, file.FrameSize());

    // Wrap around to first frame
    file.ReadNextFrame(frame.data());
    this->ComapreFrames(frame.data(), this->vfr_frames_[0][0].data, file.FrameSize());
}


}  // namespace dali
