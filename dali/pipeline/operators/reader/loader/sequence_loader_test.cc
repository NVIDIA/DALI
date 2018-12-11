// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/operators/reader/loader/sequence_loader.h"
#include "dali/test/dali_test.h"

namespace dali {

namespace {

std::string print_frame_num(int i) {
  std::stringstream ss;
  ss << std::setfill('0') << std::setw(5) << i;
  return ss.str();
}

}  // namespace

TEST(GatherExtractedStreamsTest, LoadTestDir) {
  const auto frames_dir = image_folder + "/frames";
  auto result = filesystem::GatherExtractedStreams(frames_dir);
  ASSERT_EQ(result.size(), 2);
  const auto first_stream = frames_dir + "/0/";
  ASSERT_EQ(result[0].first, first_stream);
  ASSERT_EQ(result[0].second.size(), 16);
  for (int i = 0; i < 16; i++) {
    ASSERT_EQ(result[0].second[i], first_stream + print_frame_num(i + 1) + ".png");
  }

  const auto second_stream = frames_dir + "/1/";
  ASSERT_EQ(result[1].first, second_stream);
  ASSERT_EQ(result[1].second.size(), 16);
  for (int i = 0; i < 16; i++) {
    ASSERT_EQ(result[1].second[i], second_stream + print_frame_num(i + 1) + ".png");
  }
}

TEST(CalculateSequencesCountsTest, Test) {
  std::vector<filesystem::Stream> zero_stream = {{"/0", {}}};
  auto zero_length_1 = detail::CalculateSequencesCounts(zero_stream, 1);
  ASSERT_EQ(zero_length_1[0], 0);
  auto zero_length_2 = detail::CalculateSequencesCounts(zero_stream, 2);
  ASSERT_EQ(zero_length_2[0], 0);


  std::vector<filesystem::Stream> test_streams = {
      {"/0", {"/0/00.png", "/0/01.png", "/0/02.png", "/0/03.png", "/0/04.png", "/0/05.png"}},
      {"/1", {"/1/00.png", "/1/01.png", "/1/02.png", "/1/03.png"}}};
  auto length_1 = detail::CalculateSequencesCounts(test_streams, 1);
  ASSERT_EQ(length_1[0], 6);
  ASSERT_EQ(length_1[1], 4);
  auto length_2 = detail::CalculateSequencesCounts(test_streams, 2);
  ASSERT_EQ(length_2[0], 5);
  ASSERT_EQ(length_2[1], 3);
  auto length_3 = detail::CalculateSequencesCounts(test_streams, 3);
  ASSERT_EQ(length_3[0], 4);
  ASSERT_EQ(length_3[1], 2);
  auto length_4 = detail::CalculateSequencesCounts(test_streams, 4);
  ASSERT_EQ(length_4[0], 3);
  ASSERT_EQ(length_4[1], 1);
  auto length_5 = detail::CalculateSequencesCounts(test_streams, 5);
  ASSERT_EQ(length_5[0], 2);
  ASSERT_EQ(length_5[1], 0);
  auto length_6 = detail::CalculateSequencesCounts(test_streams, 6);
  ASSERT_EQ(length_6[0], 1);
  ASSERT_EQ(length_6[1], 0);
  auto length_7 = detail::CalculateSequencesCounts(test_streams, 7);
  ASSERT_EQ(length_7[0], 0);
  ASSERT_EQ(length_7[1], 0);
}

}  // namespace dali
