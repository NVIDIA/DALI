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
#include "dali/imgcodec/parsers/jpeg2000.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"

namespace dali {
namespace imgcodec {
namespace test {

class Jpeg2000ParserTest : public ::testing::Test {
 private:
  Jpeg2000Parser parser_;
  const std::string jp2_directory_ = testing::dali_extra_path() + "/db/single/jpeg2k/";

 public:
  Jpeg2000ParserTest() {}

  void TestInvalidImageData(const std::vector<uint8_t> &data) {
    auto img = ImageSource::FromHostMem(data.data(), data.size());
    EXPECT_FALSE(parser_.CanParse(&img));
  }

  void TestValidImageHeader(const std::vector<uint8_t> &data) {
    auto img = ImageSource::FromHostMem(data.data(), data.size());
    EXPECT_TRUE(parser_.CanParse(&img));
  }

  void TestValidImage(const std::string &filename, TensorShape<> expected_shape) {
    auto img = ImageSource::FromFilename(jp2_directory_ + filename);
    EXPECT_TRUE(parser_.CanParse(&img));
    EXPECT_EQ(parser_.GetInfo(&img).shape, expected_shape);
  }
};

TEST_F(Jpeg2000ParserTest, ValidRGB) {
  TestValidImage("0/cat-1046544_640-16bit.jp2", {475, 640, 3});
  TestValidImage("0/cat-1046544_640.jp2", {475, 640, 3});
  TestValidImage("1/cat-3449999_640.jp2", {426, 640, 3});
  TestValidImage("2/tiled-cat-3113513_640.jp2", {299, 640, 3});
}

TEST_F(Jpeg2000ParserTest, UnexpectedEnd) {
  TestInvalidImageData({0, 0, 0, 7, 'j', 'P', ' '});
}

TEST_F(Jpeg2000ParserTest, InvalidType) {
  TestValidImageHeader({0, 0, 0, 8, 'j', 'P', ' ', ' '});
  TestInvalidImageData({0, 0, 0, 8, 'j', 'P', ' ', '_'});
  TestInvalidImageData({0, 0, 0, 8, 'j', 'p', ' ', ' '});
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
