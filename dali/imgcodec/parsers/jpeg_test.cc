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
#include "dali/imgcodec/parsers/jpeg.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"

namespace dali {
namespace imgcodec {
namespace test {

class JpegParserTest : public ::testing::Test {
 public:
  JpegParserTest() : parser_() {
    auto filename = testing::dali_extra_path() + "/db/single/jpeg/100/swan-3584559_640.jpg";
    img_ = ImageSource::FromFilename(filename);
    auto stream = img_.Open();
    valid_jpeg_.resize(stream->Size());
    size_t offset = 0;
    size_t n;
    while ((n = stream->Read(valid_jpeg_.data() + offset, 4096))) {
      offset += n;
    }
  }

  bool CanParse(const std::vector<uint8_t> &data) {
    auto src = ImageSource::FromHostMem(data.data(), data.size());
    return parser_.CanParse(&src);
  }

  ImageInfo GetInfo(const std::vector<uint8_t> &data) {
    auto src = ImageSource::FromHostMem(data.data(), data.size());
    return parser_.GetInfo(&src);
  }

  std::vector<uint8_t> replace(const std::vector<uint8_t> &data,
                               const std::vector<uint8_t> &old_value,
                               const std::vector<uint8_t> &new_value) {
    std::vector<uint8_t> result;
    result.reserve(data.size());
    auto it = data.begin();
    size_t n = old_value.size();
    while (it != data.end()) {
      if (it + n <= data.end() && std::equal(it, it + n, old_value.begin(), old_value.end())) {
        result.insert(result.end(), new_value.begin(), new_value.end());
        it += n;
      } else {
        result.push_back(*(it++));
      }
    }
    return result;
  }

  JpegParser parser_;
  std::vector<uint8_t> valid_jpeg_;
  ImageSource img_;
};

TEST_F(JpegParserTest, ValidJpeg) {
  EXPECT_TRUE(CanParse(valid_jpeg_));
  EXPECT_EQ(TensorShape<>(408, 640, 3), GetInfo(valid_jpeg_).shape);
}

TEST_F(JpegParserTest, FromFilename) {
  EXPECT_TRUE(parser_.CanParse(&img_));
  EXPECT_EQ(TensorShape<>(408, 640, 3), parser_.GetInfo(&img_).shape);
}

TEST_F(JpegParserTest, Empty) {
  EXPECT_FALSE(CanParse({}));
}

TEST_F(JpegParserTest, BadSoi) {
  auto bad = valid_jpeg_;
  EXPECT_EQ(0xd8, valid_jpeg_[1]);  // A valid JPEG starts with ff d8 (Start Of Image marker)...
  bad[1] = 0xc0;                    // ...but we make it ff c0, which is Start Of Frame
  EXPECT_FALSE(CanParse(bad));
}

TEST_F(JpegParserTest, NoSof) {
  // We change Start Of Frame marker into a Comment marker
  auto bad = replace(valid_jpeg_, {0xff, 0xc0}, {0xff, 0xfe});
  EXPECT_ANY_THROW(GetInfo(bad));
}

TEST_F(JpegParserTest, Padding) {
  /* https://www.w3.org/Graphics/JPEG/itu-t81.pdf section B.1.1.2 Markers
   * Any marker may optionally be preceded by any number of fill bytes,
   * which are bytes assigned code X’FF’ */
  auto padded = replace(valid_jpeg_, {0xff, 0xe0}, {0xff, 0xff, 0xff, 0xff, 0xe0});
  padded = replace(padded, {0xff, 0xe1}, {0xff, 0xff, 0xe1});
  padded = replace(padded, {0xff, 0xdb}, {0xff, 0xff, 0xff, 0xdb});
  padded = replace(padded, {0xff, 0xc0}, {0xff, 0xff, 0xff, 0xff, 0xff, 0xc0});
  EXPECT_TRUE(CanParse(padded));
  EXPECT_EQ(TensorShape<>(408, 640, 3), GetInfo(padded).shape);
}

class JpegParserOrientationTest : public ::testing::Test {
 public:
  JpegParserOrientationTest() : parser_() {}

  ImageSource GetImage(const std::string& orientation) {
    auto base = testing::dali_extra_path() + "/db/imgcodec/jpeg/orientation/padlock-406986_640_";
    auto filename = base + orientation + ".jpg";
    return ImageSource::FromFilename(filename);
  }

  int NormalizeAngle(int degrees) {
    return (degrees % 360 + 360) % 360;
  }

  JpegParser parser_;
};

TEST_F(JpegParserOrientationTest, NoOrientation) {
  auto img = GetImage("no_orientation");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(0, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(false, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}

TEST_F(JpegParserOrientationTest, NoExif) {
  auto img = GetImage("no_exif");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(0, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(false, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}


TEST_F(JpegParserOrientationTest, Horizontal) {
  auto img = GetImage("horizontal");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(0, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(false, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}

TEST_F(JpegParserOrientationTest, MirrorHorizontal) {
  auto img = GetImage("mirror_horizontal");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(0, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(true, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}

TEST_F(JpegParserOrientationTest, Rotate180) {
  auto img = GetImage("rotate_180");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(180, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(false, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}

TEST_F(JpegParserOrientationTest, MirrorVertical) {
  auto img = GetImage("mirror_vertical");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(0, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(false, orientation.flip_x);
  EXPECT_EQ(true, orientation.flip_y);
}

TEST_F(JpegParserOrientationTest, MirrorHorizontalRotate270) {
  auto img = GetImage("mirror_horizontal_rotate_270");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(360 - 270, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(false, orientation.flip_x);
  EXPECT_EQ(true, orientation.flip_y);
}

TEST_F(JpegParserOrientationTest, Rotate90) {
  auto img = GetImage("rotate_90");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(360 - 90, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(false, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}

TEST_F(JpegParserOrientationTest, MirrorHorizontalRotate90) {
  auto img = GetImage("mirror_horizontal_rotate_90");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(360 - 90, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(false, orientation.flip_x);
  EXPECT_EQ(true, orientation.flip_y);
}

TEST_F(JpegParserOrientationTest, Rotate270) {
  auto img = GetImage("rotate_270");
  auto orientation = parser_.GetInfo(&img).orientation;
  EXPECT_EQ(360 - 270, NormalizeAngle(orientation.rotate));
  EXPECT_EQ(false, orientation.flip_x);
  EXPECT_EQ(false, orientation.flip_y);
}


}  // namespace test
}  // namespace imgcodec
}  // namespace dali
