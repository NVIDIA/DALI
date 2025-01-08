// Copyright (c) 2022, 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <fstream>
#include <memory>
#include <utility>
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
        {60,  4,  3,  5, 0, 0},
        {50,  4,  3,  4, 0, 2},
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
  for (auto &td : test_data) {
    auto [num_full_batches, num_full_sequences, frames_in_last_sequence] =
            detail::DetermineBatchOutline(td.num_frames, td.frames_per_sequence, td.batch_size);
    EXPECT_EQ(num_full_batches, td.num_full_batches) << "Error in data: " << td;
    EXPECT_EQ(num_full_sequences, td.num_full_sequences) << "Error in data: " << td;
    EXPECT_EQ(frames_in_last_sequence, td.frames_in_last_sequence) << "Error in data: " << td;
  }
}


namespace {

/**
 * Load the test file as a binary string.
 */
std::string LoadTestFile(const std::string &test_file_path) {
  std::ifstream fin(test_file_path, std::ios::binary);
  if (!fin)
    throw std::runtime_error(std::string("Failed to open model file: ") + test_file_path);
  std::stringstream ss;
  ss << fin.rdbuf();
  return ss.str();
}

}  // namespace


template<typename VideoInputBackend>
class VideoInputNextOutputDataIdTest : public ::testing::Test {
 protected:
  struct TestFileDescriptor {
    std::string file_name;  /// Full path of the file.
    int n_frames;           /// Number of frames in the file.
    std::string data_id;    /// DataId used for this file.
  };

  static constexpr bool is_cpu = std::is_same_v<VideoInputBackend, CPUBackend>;


  void SetUp() final {
    CreateAndSerializePipeline();
  }


  /**
   * Performs a test on a single test file.
   * @params test_file_idx @see TestFileDescriptor
   */
  void DoTest(daliPipelineHandle *h, int test_file_idx) {
    // First, load the data to the pipeline.
    FeedExternalInput(h, LoadTestFile(test_files_[test_file_idx].file_name),
                      test_files_[test_file_idx].data_id);

    // Determine, how many iterations the input test file should produce.
    auto [num_full_batches, num_full_sequences, frames_in_last_sequence] =
            detail::DetermineBatchOutline(
                    test_files_[test_file_idx].n_frames, frames_per_sequence_, batch_size_);
    auto num_iterations_per_input =
            num_full_batches + (num_full_sequences > 0 || frames_in_last_sequence > 0 ? 1 : 0);

    // Run the pipeline to test it.
    for (int i = 0;
         // `-1`, since the last iteration will carry different result,
         // so it will be checked outside the loop.
         i < num_iterations_per_input - 1;
         i++) {
      daliRun(h);
      daliOutput(h);

      AssertDataIdTraceExist(h, i, test_file_idx);
      CheckDataIdTraceValue(h, i, test_file_idx);
      AssertDepletedTraceExists(h);
      CheckDepletedTraceValue(h, false);
    }
    /*
     * The last iteration of the pipeline shall carry a different result.
     * Since this function tests a single file, after the last iteration there shouldn't
     * be a "next_output_data_id" trace available.
     * "depleted" trace shall always be available.
     */
    daliRun(h);
    daliOutput(h);
    EXPECT_EQ(daliHasOperatorTrace(h, video_input_name_.c_str(), data_id_trace_name_.c_str()), 0);
    AssertDepletedTraceExists(h);
    CheckDepletedTraceValue(h, true);
  }


  void CreateAndSerializePipeline() {
    auto pipeline = std::make_unique<Pipeline>(batch_size_, num_threads_, device_id_);
    string device = is_cpu ? "cpu" : "mixed";
    auto storage_device = is_cpu ? StorageDevice::CPU : StorageDevice::GPU;
    pipeline->AddOperator(
            OpSpec("experimental__inputs__Video")
                    .AddArg("sequence_length", frames_per_sequence_)
                    .AddArg("device", device)
                    .AddArg("name", video_input_name_)
                    .AddOutput(video_input_name_, storage_device),
            video_input_name_);

    std::vector<std::pair<std::string, std::string>> outputs = {
            {video_input_name_, is_cpu ? "cpu" : "gpu"},
    };

    pipeline->SetOutputDescs(outputs);

    serialized_pipeline_ = pipeline->SerializeToProtobuf();
  }


  void FeedExternalInput(daliPipelineHandle *h, const std::string &encoded_video,
                         const std::string &data_id) {
    daliSetExternalInputBatchSize(h, video_input_name_.c_str(), 1);
    if (!data_id.empty()) {
      daliSetExternalInputDataId(h, video_input_name_.c_str(), data_id.data());
    }
    int64_t shapes[] = {static_cast<int64_t>(encoded_video.length())};
    daliSetExternalInput(h, video_input_name_.c_str(), device_type_t::CPU, encoded_video.data(),
                         dali_data_type_t::DALI_UINT8, shapes, 1, nullptr, DALI_ext_force_copy);
  }


  /**
   * Check, if the "next_output_data_id" trace exists, provided it should.
   */
  void AssertDataIdTraceExist(daliPipelineHandle *h, int iteration_idx, int test_file_idx) {
    bool has_data_id = daliHasOperatorTrace(h, video_input_name_.c_str(),
                                            data_id_trace_name_.c_str());
    ASSERT_EQ(
            has_data_id,
            !test_files_[test_file_idx].data_id.empty())
                          << "Failed at iteration " << iteration_idx << " of file with index "
                          << test_file_idx;
  }


  /**
   * Verify, if the "next_output_data_id" trace has a correct value (provided it should exist).
   */
  void CheckDataIdTraceValue(daliPipelineHandle *h, int iteration_idx, int test_file_idx) {
    bool has_data_id = daliHasOperatorTrace(h, video_input_name_.c_str(),
                                            data_id_trace_name_.c_str());
    if (has_data_id) {
      EXPECT_STREQ(
              daliGetOperatorTrace(h, video_input_name_.c_str(), data_id_trace_name_.c_str()),
              test_files_[test_file_idx].data_id.c_str())
              << "Failed at iteration " << iteration_idx << " of file with index " << test_file_idx;
    }
  }


  /**
   * Check, if the "depleted" trace exists.
   */
  void AssertDepletedTraceExists(daliPipelineHandle *h) {
    // The "depleted" trace should always exist.
    ASSERT_TRUE(daliHasOperatorTrace(h, video_input_name_.c_str(), depleted_trace_name_.c_str()));
  }


  /**
   * Verify the value of "depleted" trace.
   * @param shall_be_depleted Expected value.
   */
  void CheckDepletedTraceValue(daliPipelineHandle *h, bool shall_be_depleted) {
    EXPECT_STREQ(daliGetOperatorTrace(h, video_input_name_.c_str(), depleted_trace_name_.c_str()),
                 shall_be_depleted ? "true" : "false");
  }


  const int batch_size_ = 3;
  const int num_threads_ = 2;
  const int device_id_ = 0;
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
          {
                  make_string(testing::dali_extra_path(), "/db/video/cfr/test_2.mp4"),
                  60,
                  ""  // No data_id for this file.
          },
  };
  const int frames_per_sequence_ = 4;
  const std::string data_id_trace_name_ = "next_output_data_id";
  const std::string depleted_trace_name_ = "depleted";

  std::string serialized_pipeline_;
};


using VideoInputTestTypes = ::testing::Types<CPUBackend, MixedBackend>;
TYPED_TEST_SUITE(VideoInputNextOutputDataIdTest, VideoInputTestTypes);


/**
 * Tests the situation, when user provides only one file at the input to VideoInput.
 * The test file will not be split uniformly across output batches.
 */
TYPED_TEST(VideoInputNextOutputDataIdTest, VideoInputNextOutputDataIdTest) {
  daliPipelineHandle h;
  daliDeserializeDefault(&h, this->serialized_pipeline_.c_str(),
                         static_cast<int>(this->serialized_pipeline_.length()));
  this->DoTest(&h, 0);
  daliDeletePipeline(&h);
}


/**
 * Tests the situation, when user provides only one file at the input to VideoInput.
 * The test file will be split uniformly across output batches.
 */
TYPED_TEST(VideoInputNextOutputDataIdTest, OneInputFileSplitUniformlyTest) {
  daliPipelineHandle h;
  daliDeserializeDefault(&h, this->serialized_pipeline_.c_str(),
                         static_cast<int>(this->serialized_pipeline_.length()));
  this->DoTest(&h, 1);
  daliDeletePipeline(&h);
}


/**
 * Tests the situation, when user provides only one file at the input to VideoInput.
 * The test file will be split uniformly across output batches.
 * The test file does not have data_id assigned.
 */
TYPED_TEST(VideoInputNextOutputDataIdTest, OneInputFileSplitUniformlyNoDataIdTest) {
  daliPipelineHandle h;
  daliDeserializeDefault(&h, this->serialized_pipeline_.c_str(),
                         static_cast<int>(this->serialized_pipeline_.length()));
  this->DoTest(&h, 2);
  daliDeletePipeline(&h);
}


/**
 * Tests the situation, when user provides sequentially two files at the input to VideoInput.
 */
TYPED_TEST(VideoInputNextOutputDataIdTest, TwoInputFilesSeparatedTest) {
  daliPipelineHandle h;
  daliDeserializeDefault(&h, this->serialized_pipeline_.c_str(),
                         static_cast<int>(this->serialized_pipeline_.length()));
  this->DoTest(&h, 0);
  this->DoTest(&h, 2);
  daliDeletePipeline(&h);
}


/**
 * Tests the situation, when user provides sequentially two files at the input to VideoInput.
 */
TYPED_TEST(VideoInputNextOutputDataIdTest, TwoInputFilesSeparatedTest2) {
  daliPipelineHandle h;
  daliDeserializeDefault(&h, this->serialized_pipeline_.c_str(),
                         static_cast<int>(this->serialized_pipeline_.length()));
  this->DoTest(&h, 2);
  this->DoTest(&h, 1);
  daliDeletePipeline(&h);
}


/**
 * Tests the situation, when user provides multiple files. Some have data_id, some don't.
 */
TYPED_TEST(VideoInputNextOutputDataIdTest, MultipleFilesSeparatedTest) {
  daliPipelineHandle h;
  daliDeserializeDefault(&h, this->serialized_pipeline_.c_str(),
                         static_cast<int>(this->serialized_pipeline_.length()));
  this->DoTest(&h, 2);
  this->DoTest(&h, 1);
  this->DoTest(&h, 2);
  this->DoTest(&h, 0);
  this->DoTest(&h, 2);
  daliDeletePipeline(&h);
}


/**
 * Tests the situation, when user provides multiple files to the input and then runs all of them.
 */
TYPED_TEST(VideoInputNextOutputDataIdTest, MultipleInputFilesParallelTest) {
  daliPipelineHandle h;
  daliDeserializeDefault(&h, this->serialized_pipeline_.c_str(),
                         static_cast<int>(this->serialized_pipeline_.length()));

  std::vector<int> test_files_order = {2, 1, 2, 0, 2};
  std::vector<int> num_iterations_per_input(this->test_files_.size());

  // Determine the number of iteration for every test file.
  for (size_t i = 0; i < this->test_files_.size(); i++) {
    auto [num_full_batches, num_full_sequences, frames_in_last_sequence] =
            detail::DetermineBatchOutline(this->test_files_[i].n_frames, this->frames_per_sequence_,
                                          this->batch_size_);
    num_iterations_per_input[i] =
            num_full_batches + (num_full_sequences > 0 || frames_in_last_sequence > 0 ? 1 : 0);
  }

  // Feed the pipeline with the test files.
  for (const auto &tf : test_files_order) {
    this->FeedExternalInput(&h, LoadTestFile(this->test_files_[tf].file_name),
                            this->test_files_[tf].data_id);
  }

  // Run test or almost all the test files (except the last one).
  for (size_t i = 0; i < test_files_order.size() - 1; i++) {
    const auto &test_file_idx = test_files_order[i];
    const auto &next_test_file_idx = test_files_order[i + 1];
    for (int iteration_idx = 0;
         iteration_idx < num_iterations_per_input[test_file_idx] - 1; iteration_idx++) {
      daliRun(&h);
      daliOutput(&h);
      this->AssertDataIdTraceExist(&h, iteration_idx, test_file_idx);
      this->CheckDataIdTraceValue(&h, iteration_idx, test_file_idx);
    }
    daliRun(&h);
    daliOutput(&h);
    this->AssertDataIdTraceExist(&h, num_iterations_per_input[test_file_idx] - 1,
                                 next_test_file_idx);
    this->CheckDataIdTraceValue(&h, num_iterations_per_input[test_file_idx] - 1,
                                next_test_file_idx);
  }
  // The last test file should just clear the "next_output_data_id" trace after it's done.
  auto test_file_idx = test_files_order.back();
  for (int iteration_idx = 0;
       iteration_idx < num_iterations_per_input[test_file_idx] - 1; iteration_idx++) {
    daliRun(&h);
    daliOutput(&h);
    this->AssertDataIdTraceExist(&h, iteration_idx, test_file_idx);
    this->CheckDataIdTraceValue(&h, iteration_idx, test_file_idx);
  }
  daliRun(&h);
  daliOutput(&h);
  EXPECT_EQ(daliHasOperatorTrace(&h, this->video_input_name_.c_str(),
                                 this->data_id_trace_name_.c_str()), 0);
  daliDeletePipeline(&h);
}

}  // namespace dali::test
