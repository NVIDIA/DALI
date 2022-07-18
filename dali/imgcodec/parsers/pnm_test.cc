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

#include <string>
#include <vector>
#include "dali/imgcodec/image_source.h"
#include "dali/imgcodec/parsers/pnm.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"

namespace dali {
namespace imgcodec {
namespace test {

class PnmParserTest : public ::testing::Test {
 public:
  PnmParserTest() {}

  bool CanParse(std::vector<char> data) {
    auto src = ImageSource::FromHostMem(data.data(), data.size());
    return parser_.CanParse(&src);
  }

  PnmParser parser_;
  const std::string pnm_directory_ = testing::dali_extra_path() + "/db/single/pnm/0/";
};

TEST_F(PnmParserTest, ValidPbm) {
  auto img = ImageSource::FromFilename(pnm_directory_ + "cat-2184682_640.pbm");
  EXPECT_TRUE(parser_.CanParse(&img));
  TensorShape<> expected = {398, 640, 1};
  EXPECT_EQ(parser_.GetInfo(&img).shape, expected);
}

TEST_F(PnmParserTest, ValidPgm) {
  auto img = ImageSource::FromFilename(pnm_directory_ + "cat-1245673_640.pgm");
  EXPECT_TRUE(parser_.CanParse(&img));
  TensorShape<> expected = {423, 640, 1};
  EXPECT_EQ(parser_.GetInfo(&img).shape, expected);
}

TEST_F(PnmParserTest, ValidPpm) {
  auto img = ImageSource::FromFilename(pnm_directory_ + "cat-111793_640.ppm");
  EXPECT_TRUE(parser_.CanParse(&img));
  TensorShape<> expected = {426, 640, 3};
  EXPECT_EQ(parser_.GetInfo(&img).shape, expected);
}

TEST_F(PnmParserTest, ValidPbmComment) {
  const char data[] =
      "P1\n"
      "#This is an example bitmap of the letter \"J\"\n"
      "6 10\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "1 0 0 0 1 0\n"
      "0 1 1 1 0 0\n"
      "0 0 0 0 0 0\n"
      "0 0 0 0 0 0\n";
  auto img = ImageSource::FromHostMem(data, sizeof(data));
  EXPECT_TRUE(parser_.CanParse(&img));
  TensorShape<> expected = {10, 6, 1};
  EXPECT_EQ(parser_.GetInfo(&img).shape, expected);
}

TEST_F(PnmParserTest, ValidPbmCommentInsideToken) {
  const char data[] =
      "P1\n"
      "6 1#Comment can be inside of a token\n"
      "0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "1 0 0 0 1 0\n"
      "0 1 1 1 0 0\n"
      "0 0 0 0 0 0\n"
      "0 0 0 0 0 0\n";
  auto img = ImageSource::FromHostMem(data, sizeof(data));
  EXPECT_TRUE(parser_.CanParse(&img));
  TensorShape<> expected = {10, 6, 1};
  EXPECT_EQ(parser_.GetInfo(&img).shape, expected);
}

TEST_F(PnmParserTest, ValidPbmCommentInsideWhitespaces) {
  const char data[] =
      "P1 \n"
      "#Comment can be inside of whitespaces\n"
      " 6 10\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "1 0 0 0 1 0\n"
      "0 1 1 1 0 0\n"
      "0 0 0 0 0 0\n"
      "0 0 0 0 0 0\n";
  auto img = ImageSource::FromHostMem(data, sizeof(data));
  EXPECT_TRUE(parser_.CanParse(&img));
  TensorShape<> expected = {10, 6, 1};
  EXPECT_EQ(parser_.GetInfo(&img).shape, expected);
}

TEST_F(PnmParserTest, PamFormat) {
  EXPECT_FALSE(CanParse({'P', '7', ' '}));
}

TEST_F(PnmParserTest, WhitespacesTest) {
  for (const auto whitespace : {' ', '\n', '\f', '\r', '\t', '\v'})
    EXPECT_TRUE(CanParse({'P', '6', whitespace}));
}

TEST_F(PnmParserTest, LowercaseP) {
  EXPECT_FALSE(CanParse({'p', '6', ' '}));
}

TEST_F(PnmParserTest, MissingWhitespace) {
  EXPECT_FALSE(CanParse({'P', '6', '1'}));
}

TEST_F(PnmParserTest, Empty) {
  EXPECT_FALSE(CanParse({}));
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
