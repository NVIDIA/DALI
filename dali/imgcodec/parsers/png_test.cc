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

#include "dali/imgcodec/parsers/png.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"

namespace dali {
namespace imgcodec {
namespace test {

class PngParserTest : public ::testing::Test {
 public:
  PngParserTest() : parser_() {
    auto filename = testing::dali_extra_path() + "/db/single/png/0/domestic-cat-726989_640.png";
    img_ = ImageSource::FromFilename(filename);
    auto stream = img_.Open();
    valid_png_.resize(stream->Size());
    size_t offset = 0;
    size_t n;
    while ((n = stream->Read(valid_png_.data() + offset, 4096))) {
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

  PngParser parser_;
  std::vector<char> valid_png_;
  ImageSource img_;
};

TEST_F(PngParserTest, ValidPng) {
  EXPECT_TRUE(CanParse(valid_png_));
}

TEST_F(PngParserTest, BadMagic) {
  auto bad = valid_png_;
  std::swap(bad[2], bad[3]);
  EXPECT_FALSE(CanParse(bad));
}

TEST_F(PngParserTest, Empty) {
  EXPECT_FALSE(CanParse({}));
}

TEST_F(PngParserTest, FromFilename) {
  EXPECT_TRUE(parser_.CanParse(&img_));
}

TEST_F(PngParserTest, CheckInfo) {
  auto info = GetInfo(valid_png_);
  TensorShape<> expected_shape = {426, 640, 3};
  ASSERT_EQ(info.shape, expected_shape);
}

class PngParserOrientationTest : public ::testing::Test {
 public:
  PngParserOrientationTest() : parser_() {}

  ImageSource GetImage(const std::string& orientation) {
    auto base = testing::dali_extra_path() + "/db/imgcodec/png/orientation/bicycle-161524_1280_";
    auto filename = base + orientation + ".png";
    return ImageSource::FromFilename(filename);
  }

  int NormalizeAngle(int degrees) {
    return (degrees % 360 + 360) % 360;
  }

  PngParser parser_;
};

TEST_F(PngParserOrientationTest, NoOrientation) {
  auto img = GetImage("no_orientation");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(0, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(false, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}

TEST_F(PngParserOrientationTest, Horizontal) {
  auto img = GetImage("horizontal");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(0, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(false, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}

TEST_F(PngParserOrientationTest, MirrorHorizontal) {
  auto img = GetImage("mirror_horizontal");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(0, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(true, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}

TEST_F(PngParserOrientationTest, Rotate180) {
  auto img = GetImage("rotate_180");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(180, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(false, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}

TEST_F(PngParserOrientationTest, MirrorVertical) {
  auto img = GetImage("mirror_vertical");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(0, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(false, orientation.flip_x);
  EXPECT_EQ(true, orientation.flip_y);
}

TEST_F(PngParserOrientationTest, MirrorHorizontalRotate270) {
  auto img = GetImage("mirror_horizontal_rotate_270");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(360 - 270, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(true, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}

TEST_F(PngParserOrientationTest, Rotate90) {
  auto img = GetImage("rotate_90");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(360 - 90, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(false, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}

TEST_F(PngParserOrientationTest, MirrorHorizontalRotate90) {
  auto img = GetImage("mirror_horizontal_rotate_90");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(360 - 90, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(true, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}

TEST_F(PngParserOrientationTest, Rotate270) {
  auto img = GetImage("rotate_270");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(360 - 270, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(false, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
