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
#include "dali/imgcodec/parsers/webp.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"

namespace dali {
namespace imgcodec {
namespace test {

class WebpParserTest : public ::testing::Test {
 private:
  WebpParser parser_;
  const std::string webp_directory_ = testing::dali_extra_path() + "/db/single/webp/";

 public:
  std::vector<uint8_t> ReadFile(const std::string &filename) {
    std::ifstream stream(webp_directory_ + filename, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
  }

  void TestInvalidImageData(const std::vector<uint8_t> &data) {
    auto img = ImageSource::FromHostMem(data.data(), data.size());
    EXPECT_FALSE(parser_.CanParse(&img));
  }

  void TestValidImage(const std::string &filename, TensorShape<> expected_shape) {
    auto img = ImageSource::FromFilename(webp_directory_ + filename);
    EXPECT_TRUE(parser_.CanParse(&img));
    EXPECT_EQ(parser_.GetInfo(&img).shape, expected_shape);
  }
};

TEST_F(WebpParserTest, ValidWebpLossy) {
  TestValidImage("lossy/kitty-2948404_640.webp", {433, 640, 3});
  TestValidImage("lossy/domestic-cat-726989_640.webp", {426, 640, 3});
  TestValidImage("lossy/cat-300572_640.webp", {536, 640, 3});
}

TEST_F(WebpParserTest, ValidWebpLossless) {
  TestValidImage("lossless/kitty-2948404_640.webp", {433, 640, 3});
  TestValidImage("lossless/domestic-cat-726989_640.webp", {426, 640, 3});
  TestValidImage("lossless/cat-300572_640.webp", {536, 640, 3});
}

TEST_F(WebpParserTest, ValidWebpLosslessAlpha) {
  TestValidImage("lossless-alpha/camel-1987672_640.webp", {426, 640, 4});
  TestValidImage("lossless-alpha/elephant-3095555_640.webp", {512, 640, 4});
}

TEST_F(WebpParserTest, InvalidRiffHeader) {
  for (const std::string subdir : {"lossy/", "lossless/"}) {
    const std::string filename = subdir + "cat-1046544_640.webp";
    const auto image_data = ReadFile(filename);

    {
      auto data = image_data;
      data[2] = 'f';
      TestInvalidImageData(data);
    }
    {
      auto data = image_data;
      data[8] = 'w';
      TestInvalidImageData(data);
    }
  }
}

TEST_F(WebpParserTest, ExtendedFileFormat) {
  auto image_data = ReadFile("lossy/cat-3449999_640.webp");
  image_data[15] = 'X';
  TestInvalidImageData(image_data);
}

TEST_F(WebpParserTest, InvalidVp8Identifier) {
  const auto image_data = ReadFile("lossy/cat-111793_640.webp");
  {
    auto data = image_data;
    data[15] = 0;
    TestInvalidImageData(data);
  }
  {
    auto data = image_data;
    data[12] = 'v';
    TestInvalidImageData(data);
  }
  {
    auto data = image_data;
    data[14] = '0';
    TestInvalidImageData(data);
  }
}

TEST_F(WebpParserTest, InvalidSyncCodeLossy) {
  const auto image_data = ReadFile("lossy/cat-3449999_640.webp");
  for (size_t i = 0; i < 3; i++) {
    auto data = image_data;
    data[23 + i]++;
    TestInvalidImageData(data);
  }
}

TEST_F(WebpParserTest, InvalidSyncCodeLossless) {
  auto image_data = ReadFile("lossless/cat-3504008_640.webp");
  image_data[20]++;
  TestInvalidImageData(image_data);
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
