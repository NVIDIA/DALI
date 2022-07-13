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

#include <vector>
#include <string>

#include "dali/imgcodec/image_source.h"
#include "dali/imgcodec/parsers/tiff.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"

namespace dali {
namespace imgcodec {
namespace test {

class TiffParserTest : public ::testing::Test {
 public:
  TiffParserTest() : parser_() {
    auto filename = testing::dali_extra_path() + "/db/single/tiff/0/cat-1245673_640.tiff";
    img_ = ImageSource::FromFilename(filename);
    auto stream = img_.Open();
    valid_tiff_.resize(stream->Size());
    size_t offset = 0;
    size_t n;
    while ((n = stream->Read(valid_tiff_.data() + offset, 4096))) {
      offset += n;
    }
  }

  bool CanParse(std::vector<char> data) {
    auto src = ImageSource::FromHostMem(data.data(), data.size());
    return parser_.CanParse(&src);
  }

  TiffParser parser_;
  std::vector<char> valid_tiff_;
  ImageSource img_;
};

TEST_F(TiffParserTest, ValidTiff) {
  EXPECT_TRUE(CanParse(valid_tiff_));
}

TEST_F(TiffParserTest, BadEndianess) {
  auto bad = valid_tiff_;
  bad[0] = 'M';
  bad[1] = 'I';
  EXPECT_FALSE(CanParse(bad));
}

TEST_F(TiffParserTest, BadMagic) {
  auto bad = valid_tiff_;
  std::swap(bad[2], bad[3]);
  EXPECT_FALSE(CanParse(bad));
}

TEST_F(TiffParserTest, IncompleteHeader) {
  EXPECT_FALSE(CanParse({'I', 'I', 0}));
}

TEST_F(TiffParserTest, Empty) {
  EXPECT_FALSE(CanParse({}));
}

TEST_F(TiffParserTest, FromFilename) {
  EXPECT_TRUE(parser_.CanParse(&img_));
}


class TiffParserOrientationTest : public ::testing::Test {
 public:
  TiffParserOrientationTest() : parser_() {}

  ImageSource GetImage(const std::string& orientation) {
    auto base = testing::dali_extra_path() + "/db/imgcodec/tiff/orientation/cat-1046544_640_";
    auto filename = base + orientation + ".tiff";
    return ImageSource::FromFilename(filename);
  }

  int NormalizeAngle(int degrees) {
    return (degrees % 360 + 360) % 360;
  }

  TiffParser parser_;
};

TEST_F(TiffParserOrientationTest, Horizontal) {
  auto img = GetImage("horizontal");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(0, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(false, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}

TEST_F(TiffParserOrientationTest, MirrorHorizontal) {
  auto img = GetImage("mirror_horizontal");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(0, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(true, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}

TEST_F(TiffParserOrientationTest, Rotate180) {
  auto img = GetImage("rotate_180");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(180, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(false, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}

TEST_F(TiffParserOrientationTest, MirrorVertical) {
  auto img = GetImage("mirror_vertical");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(0, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(false, orientation.flip_x);
  EXPECT_EQ(true, orientation.flip_y);
}

TEST_F(TiffParserOrientationTest, MirrorHorizontalRotate270) {
  auto img = GetImage("mirror_horizontal_rotate_270");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(360 - 270, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(true, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}

TEST_F(TiffParserOrientationTest, Rotate90) {
  auto img = GetImage("rotate_90");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(360 - 90, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(false, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}

TEST_F(TiffParserOrientationTest, MirrorHorizontalRotate90) {
  auto img = GetImage("mirror_horizontal_rotate_90");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(360 - 90, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(true, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}

TEST_F(TiffParserOrientationTest, Rotate270) {
  auto img = GetImage("rotate_270");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(360 - 270, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(false, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}

TEST_F(TiffParserOrientationTest, NoOrientation) {
  auto img = GetImage("no_orientation");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(0, orientation.rotate);
  EXPECT_EQ(false, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
