// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>
#include "dali/c_api.h"
#include "dali/operators/input/video_input.h"
#include "dali/pipeline/pipeline.h"
#include "dali/test/dali_test_config.h"

namespace dali::test {

namespace {
struct BatchOutlineTestData {
  int num_frames, frames_per_sequence, batch_size;
  int num_full_batches, num_full_sequences, frames_in_last_sequence;
};
std::vector<BatchOutlineTestData> test_data{
        {60,  7,  1,  8, 0, 4},
        {40,  7,  3,  1, 2, 5},
        {100, 50, 2,  1, 0, 0},
        {100, 49, 2,  1, 0, 2},
        {1,   10, 10, 0, 0, 1},
        {0,   1,  1,  0, 0, 0},
        {1,   1,  1,  1, 0, 0},
        {8,   2,  3,  1, 1, 0},
};


std::ostream &operator<<(std::ostream &os, const BatchOutlineTestData &td) {
  return os << "{" << td.num_frames << ", " << td.frames_per_sequence << ", " << td.batch_size
            << ", " << td.num_full_batches << ", " << td.num_full_sequences << ", "
            << td.frames_in_last_sequence << "}";
}
}  // namespace


TEST(VideoInputTest, DetermineBatchOutlineTest) {
  for (auto &td: test_data) {
    auto [num_full_batches, num_full_sequences, frames_in_last_sequence] =
            detail::DetermineBatchOutline(td.num_frames, td.frames_per_sequence, td.batch_size);
    EXPECT_EQ(num_full_batches, td.num_full_batches) << "Error in data: " << td;
    EXPECT_EQ(num_full_sequences, td.num_full_sequences) << "Error in data: " << td;
    EXPECT_EQ(frames_in_last_sequence, td.frames_in_last_sequence) << "Error in data: " << td;
  }
}


namespace {

std::string LoadTestFile(const std::string &test_file_path) {
  std::ifstream fin(test_file_path, std::ios::binary);
  if (!fin)
    throw std::runtime_error(std::string("Failed to open model file: ") + test_file_path);
  std::stringstream ss;
  ss << fin.rdbuf();
  return ss.str();
}

}  // namespace


class VideoInputNextOutputDataIdTest
        : public ::testing::Test {  // TODO(mszolucha) figure out a better name than DataId
 protected:
  struct TestFileDescriptor {
    std::string file_name;  /// Full path of the file.
    int n_frames;           /// Number of frames in the file.
    std::string data_id;    /// DataId used for this file.
  };


  void SetUp() final {
    CreateAndSerializePipeline();
  }


  void DoTest(daliPipelineHandle *h, int test_file_idx) {
    FeedExternalInput(h, LoadTestFile(test_files_[test_file_idx].file_name),
                      test_files_[test_file_idx].data_id);
    for (int i = 0;
         i < test_files_[test_file_idx].n_frames / frames_per_sequence_ / batch_size_;
         i++) {
      daliRun(h);
      daliOutput(h);
      ASSERT_EQ(daliHasOperatorTrace(h, video_input_name_.c_str(), trace_name_.c_str()), 0);
      EXPECT_STREQ(
              daliGetOperatorTrace(h, video_input_name_.c_str(), trace_name_.c_str()),
              test_files_[test_file_idx].data_id.c_str());
    }
    daliRun(h);
    daliOutput(h);
    EXPECT_NE(daliHasOperatorTrace(h, video_input_name_.c_str(), trace_name_.c_str()), 0);
  }


  void CreateAndSerializePipeline() {
    auto pipeline = std::make_unique<Pipeline>(batch_size_, num_threads_, device_id_);
    pipeline->AddOperator(OpSpec("experimental__inputs__Video")
                                  .AddArg("sequence_length", 5)
                                  .AddArg("device", "cpu")
                                  .AddOutput("VIDEO_OUTPUT", "cpu"),
                          video_input_name_);

    std::vector<std::pair<std::string, std::string>> outputs = {
            {"VIDEO_OUTPUT", "cpu"},
    };

    pipeline->SetOutputDescs(outputs);

    serialized_pipeline_ = pipeline->SerializeToProtobuf();
  }


  void FeedExternalInput(daliPipelineHandle *h, const std::string &encoded_video,
                         const std::string &data_id) {
    if (data_id.empty()) {
      daliSetExternalInputDataId(h, video_input_name_.c_str(), data_id.data());
    }
    int64_t shapes[] = {static_cast<int64_t>(encoded_video.length())};
    daliSetExternalInput(h, video_input_name_.c_str(), device_type_t::CPU, encoded_video.data(),
                         dali_data_type_t::DALI_UINT8, shapes, 1, nullptr, DALI_ext_default);
  }


  const int batch_size_ = 3;
  const int num_threads_ = 2;
  const int device_id_ = 0;
  const int n_iterations_ = 50;
  const std::string video_input_name_ = "VIDEO_INPUT";
  const std::vector<TestFileDescriptor> test_files_ = {
          {
                  make_string(testing::dali_extra_path(), "/db/video/cfr/test_1.mp4"),
                  50,
                  "there will be cake"
          },
          {
                  make_string(testing::dali_extra_path(), "/db/video/cfr/test_2.mp4"),
                  60,
                  "cake is a lie"
          },
  };
  const int frames_per_sequence_ = 4;
  const std::string trace_name_ = "next_output_data_id";

  std::string serialized_pipeline_;
};


/**
 * Tests the situation, when user provides only one file at the input to VideoInput.
 */
TEST_F(VideoInputNextOutputDataIdTest, OneInputFileTest) {
  daliPipelineHandle h;
  daliDeserializeDefault(&h, serialized_pipeline_.c_str(),
                         static_cast<int>(serialized_pipeline_.length()));
  DoTest(&h, 0);
}


/**
 * Tests the situation, when user provides sequentially two files at the input to VideoInput.
 */
TEST_F(VideoInputNextOutputDataIdTest, TwoInputFilesSeparatedTest) {
  daliPipelineHandle h;
  daliDeserializeDefault(&h, serialized_pipeline_.c_str(),
                         static_cast<int>(serialized_pipeline_.length()));
  DoTest(&h, 0);
  DoTest(&h, 1);
}


/**
 * Tests the situation, when user provides two files in parallel at the input to VideoInput.
 */
TEST_F(VideoInputNextOutputDataIdTest, TwoInputFilesParallelTest) {
  daliPipelineHandle h;
  daliDeserializeDefault(&h, serialized_pipeline_.c_str(),
                         static_cast<int>(serialized_pipeline_.length()));
  FeedExternalInput(&h, LoadTestFile(test_files_[0].file_name), test_files_[0].data_id);
  FeedExternalInput(&h, LoadTestFile(test_files_[1].file_name), test_files_[1].data_id);
  for (int i = 0; i < test_files_[0].n_frames / frames_per_sequence_ / batch_size_; i++) {
    daliRun(&h);
    daliOutput(&h);
    ASSERT_EQ(daliHasOperatorTrace(&h, video_input_name_.c_str(), trace_name_.c_str()), 0);
    EXPECT_STREQ(daliGetOperatorTrace(&h, video_input_name_.c_str(), trace_name_.c_str()),
                 test_files_[0].data_id.c_str());
  }
  daliRun(&h);
  daliOutput(&h);
  ASSERT_EQ(daliHasOperatorTrace(&h, video_input_name_.c_str(), trace_name_.c_str()), 0);
  EXPECT_STREQ(daliGetOperatorTrace(&h, video_input_name_.c_str(), trace_name_.c_str()),
               test_files_[1].data_id.c_str());
  for (int i = 0; i < test_files_[1].n_frames / frames_per_sequence_ / batch_size_; i++) {
    daliRun(&h);
    daliOutput(&h);
    ASSERT_EQ(daliHasOperatorTrace(&h, video_input_name_.c_str(), trace_name_.c_str()), 0);
    EXPECT_STREQ(daliGetOperatorTrace(&h, video_input_name_.c_str(), trace_name_.c_str()),
                 test_files_[1].data_id.c_str());
  }
  EXPECT_NE(daliHasOperatorTrace(&h, video_input_name_.c_str(), trace_name_.c_str()), 0);
}

}  // namespace dali::test
