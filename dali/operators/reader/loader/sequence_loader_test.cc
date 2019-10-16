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

#include "dali/operators/reader/loader/sequence_loader.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"

namespace dali {

namespace {

std::string print_frame_num(int i) {
  std::stringstream ss;
  ss << std::setfill('0') << std::setw(5) << i;
  return ss.str();
}

}  // namespace

TEST(GatherExtractedStreamsTest, LoadTestDir) {
  const auto frames_dir = testing::dali_extra_path() + "/db/sequence/frames";
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

TEST(GenerateSequencesTest, Test) {
  std::vector<filesystem::Stream> zero_stream = {{"/0", {}}};
  auto zero_length_1 = detail::GenerateSequences(zero_stream, 1, 1, 1);
  ASSERT_EQ(zero_length_1.size(), 0);

  std::vector<filesystem::Stream> test_streams = {
      {"/0", {"/0/00.png", "/0/01.png", "/0/02.png", "/0/03.png", "/0/04.png", "/0/05.png"}},
      {"/1", {"/1/00.png", "/1/01.png", "/1/02.png", "/1/03.png"}}};

  auto seq_1_1_1 = detail::GenerateSequences(test_streams, 1, 1, 1);
  auto exp_1_1_1 = std::vector<std::vector<std::string>>{
      {"/0/00.png"}, {"/0/01.png"}, {"/0/02.png"}, {"/0/03.png"}, {"/0/04.png"},
      {"/0/05.png"}, {"/1/00.png"}, {"/1/01.png"}, {"/1/02.png"}, {"/1/03.png"}};
  ASSERT_EQ(seq_1_1_1, exp_1_1_1);

  auto seq_1_2_1 = detail::GenerateSequences(test_streams, 1, 2, 1);
  auto exp_1_2_1 = std::vector<std::vector<std::string>>{
      {"/0/00.png"}, {"/0/02.png"}, {"/0/04.png"}, {"/1/00.png"}, {"/1/02.png"}};
  ASSERT_EQ(seq_1_2_1, exp_1_2_1);

  auto seq_2_1_1 = detail::GenerateSequences(test_streams, 2, 1, 1);
  auto exp_2_1_1 = std::vector<std::vector<std::string>>{
      {"/0/00.png", "/0/01.png"}, {"/0/01.png", "/0/02.png"}, {"/0/02.png", "/0/03.png"},
      {"/0/03.png", "/0/04.png"}, {"/0/04.png", "/0/05.png"}, {"/1/00.png", "/1/01.png"},
      {"/1/01.png", "/1/02.png"}, {"/1/02.png", "/1/03.png"}};
  ASSERT_EQ(seq_2_1_1, exp_2_1_1);

  auto seq_2_1_2 = detail::GenerateSequences(test_streams, 2, 1, 2);
  auto exp_2_1_2 = std::vector<std::vector<std::string>>{
      {"/0/00.png", "/0/02.png"}, {"/0/01.png", "/0/03.png"}, {"/0/02.png", "/0/04.png"},
      {"/0/03.png", "/0/05.png"}, {"/1/00.png", "/1/02.png"}, {"/1/01.png", "/1/03.png"}};
  ASSERT_EQ(seq_2_1_2, exp_2_1_2);

  auto seq_2_2_2 = detail::GenerateSequences(test_streams, 2, 2, 2);
  auto exp_2_2_2 = std::vector<std::vector<std::string>>{
      {"/0/00.png", "/0/02.png"}, {"/0/02.png", "/0/04.png"}, {"/1/00.png", "/1/02.png"}};
  ASSERT_EQ(seq_2_2_2, exp_2_2_2);
}

}  // namespace dali
