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

#include "dali/imgcodec/parsers/bmp.h"
#include <string>
#include <vector>
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"

namespace dali {
namespace imgcodec {
namespace test {

class BmpParserTest : public ::testing::Test {
 public:
  BmpParserTest() : parser_() {
    LoadFileStream("/db/single/bmp/0/cat-1245673_640.bmp");
  }

  void LoadFileStream(const std::string& filename) {
    auto filepath = testing::dali_extra_path() + filename;
    img_ = ImageSource::FromFilename(filepath);
    auto stream = img_.Open();
    valid_bmp_.resize(stream->Size());
    size_t offset = 0;
    size_t n;
    while ((n = stream->Read(valid_bmp_.data() + offset, 4096))) {
      offset += n;
    }
  }

  bool CanParse(std::vector<char> data) {
    auto src = ImageSource::FromHostMem(data.data(), data.size());
    return parser_.CanParse(&src);
  }

  ImageInfo GetInfo(std::vector<char> data) {
    auto src = ImageSource::FromHostMem(data.data(), data.size());
    return parser_.GetInfo(&src);
  }

  BmpParser parser_;
  std::vector<char> valid_bmp_;
  ImageSource img_;
};

TEST_F(BmpParserTest, ValidBmp) {
  EXPECT_TRUE(CanParse(valid_bmp_));
}

TEST_F(BmpParserTest, BadMagic) {
  auto bad = valid_bmp_;
  std::swap(bad[0], bad[1]);
  EXPECT_FALSE(CanParse(bad));
}

TEST_F(BmpParserTest, Empty) {
  EXPECT_FALSE(CanParse({}));
}

TEST_F(BmpParserTest, FromFilename) {
  EXPECT_TRUE(parser_.CanParse(&img_));
}

TEST_F(BmpParserTest, CheckInfoValidBmp) {
  auto info = GetInfo(valid_bmp_);
  TensorShape<> expected_shape = {423, 640, 3};
  ASSERT_EQ(info.shape, expected_shape);
}

TEST_F(BmpParserTest, CheckInfoValidBmpGray) {
  LoadFileStream("/db/single/bmp/0/cat-111793_640_grayscale.bmp");
  auto info = GetInfo(valid_bmp_);
  TensorShape<> expected_shape = {426, 640, 3};
  ASSERT_EQ(info.shape, expected_shape);
}

TEST_F(BmpParserTest, CheckInfoValidBmpPalette8bit) {
  LoadFileStream("/db/single/bmp/0/cat-111793_640_palette_8bit.bmp");
  auto info = GetInfo(valid_bmp_);
  TensorShape<> expected_shape = {426, 640, 3};
  ASSERT_EQ(info.shape, expected_shape);
}

TEST_F(BmpParserTest, CheckInfoValidBmpPalette1bit) {
  LoadFileStream("/db/single/bmp/0/cat-111793_640_palette_1bit.bmp");
  auto info = GetInfo(valid_bmp_);
  TensorShape<> expected_shape = {426, 640, 3};
  ASSERT_EQ(info.shape, expected_shape);
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
