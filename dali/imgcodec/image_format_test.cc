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

#include <sstream>
#include <string>
#include <vector>
#include "dali/imgcodec/image_format.h"
#include "dali/imgcodec/parsers/bmp.h"
#include "dali/imgcodec/parsers/jpeg.h"
#include "dali/imgcodec/parsers/jpeg2000.h"
#include "dali/imgcodec/parsers/png.h"
#include "dali/imgcodec/parsers/pnm.h"
#include "dali/imgcodec/parsers/tiff.h"
#include "dali/imgcodec/parsers/webp.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"

namespace dali {
namespace imgcodec {
namespace test {

class ImageFormatTest : public ::testing::Test {
 public:
  ImageFormatTest() {
    format_registry_.RegisterFormat(
        std::make_shared<ImageFormat>("jpeg", std::make_shared<JpegParser>()));
    format_registry_.RegisterFormat(
        std::make_shared<ImageFormat>("png", std::make_shared<PngParser>()));
    format_registry_.RegisterFormat(
        std::make_shared<ImageFormat>("bmp", std::make_shared<BmpParser>()));
    format_registry_.RegisterFormat(
        std::make_shared<ImageFormat>("tiff", std::make_shared<TiffParser>()));
    format_registry_.RegisterFormat(
        std::make_shared<ImageFormat>("pnm", std::make_shared<PnmParser>()));
    format_registry_.RegisterFormat(
        std::make_shared<ImageFormat>("jpeg2000", std::make_shared<Jpeg2000Parser>()));
    format_registry_.RegisterFormat(
        std::make_shared<ImageFormat>("webp", std::make_shared<WebpParser>()));
  }

  void Test(std::string filename, std::string expected_format, TensorShape<> expected_sh) {
    auto img = ImageSource::FromFilename(filename);
    auto fmt = this->format_registry_.GetImageFormat(&img);
    ASSERT_NE(fmt, nullptr);
    ASSERT_EQ(expected_format, fmt->Name());
    auto image_info = fmt->Parser()->GetInfo(&img);
    ASSERT_EQ(expected_sh, image_info.shape);
  }

  class DummyParser : public ImageParser {
   public:
    ImageInfo GetInfo(ImageSource *encoded) const override {
      return {};
    }

    bool CanParse(ImageSource *encoded) const override {
      return false;
    }

    using ImageParser::ReadHeader;
  };

  ImageFormatRegistry format_registry_;
  std::vector<char> data_;
};

TEST_F(ImageFormatTest, DISABLED_Jpeg) {
  Test(testing::dali_extra_path() + "/db/single/jpeg/372/baboon-174073_1280.jpg", "jpeg",
       TensorShape<>(720, 1280, 3));
}

TEST_F(ImageFormatTest, DISABLED_Png) {
  Test(testing::dali_extra_path() + "/db/single/png/0/cat-3504008_640.png", "png",
       TensorShape<>(425, 640, 3));
}

TEST_F(ImageFormatTest, Bmp) {
  Test(testing::dali_extra_path() + "/db/single/bmp/0/cat-1046544_640.bmp", "bmp",
       TensorShape<>(475, 640, 3));
}

TEST_F(ImageFormatTest, Tiff) {
  Test(testing::dali_extra_path() + "/db/single/tiff/0/cat-1245673_640.tiff", "tiff",
       TensorShape<>(423, 640, 3));
}

TEST_F(ImageFormatTest, DISABLED_Pnm) {
  Test(testing::dali_extra_path() + "/db/single/pnm/0/cat-300572_640.pnm", "pnm",
       TensorShape<>(536, 640, 3));
}

TEST_F(ImageFormatTest, DISABLED_Jpeg2000) {
  Test(testing::dali_extra_path() + "/db/single/jpeg2k/0/cat-3113513_640.jp2", "jpeg2000",
       TensorShape<>(299, 640, 3));
}

TEST_F(ImageFormatTest, DISABLED_Webp) {
  Test(testing::dali_extra_path() + "/db/single/webp/lossy/cat-1245673_640.webp", "webp",
       TensorShape<>(423, 640, 3));
}

TEST_F(ImageFormatTest, ReadHeaderHostMem) {
  const uint8_t data[] = {0, 1, 2, 3};
  uint8_t buffer[16];
  auto src = ImageSource::FromHostMem(data, 4);
  DummyParser p;
  EXPECT_EQ(4, p.ReadHeader(buffer, &src, 5));
  EXPECT_EQ(0, buffer[0]);
  EXPECT_EQ(1, buffer[1]);
  EXPECT_EQ(2, buffer[2]);
  EXPECT_EQ(3, buffer[3]);
}

TEST_F(ImageFormatTest, ReadHeaderStream) {
  auto src = ImageSource::FromFilename(
    testing::dali_extra_path() + "/db/single/tiff/0/cat-1245673_640.tiff");
  uint8_t buffer[4];
  DummyParser p;
  EXPECT_EQ(4, p.ReadHeader(buffer, &src, 4));
  EXPECT_EQ('I', buffer[0]);
  EXPECT_EQ('I', buffer[1]);
  EXPECT_EQ(42, buffer[2]);
  EXPECT_EQ(0, buffer[3]);
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
