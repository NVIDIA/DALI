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
#include "dali/imgcodec/formats/bmp.h"
#include "dali/imgcodec/formats/jpeg.h"
#include "dali/imgcodec/formats/jpeg2000.h"
#include "dali/imgcodec/formats/png.h"
#include "dali/imgcodec/formats/pnm.h"
#include "dali/imgcodec/formats/tiff.h"
#include "dali/imgcodec/formats/webp.h"
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

  EncodedImageHostMemory LoadImage(const char *filename) {
    std::ifstream ifs(filename, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    auto filesize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data_.resize(filesize);
    ifs.read(&data_[0], filesize);
    return EncodedImageHostMemory(&data_[0], data_.size());
  }

  void Test(std::string filename, std::string expected_format, TensorShape<> expected_sh) {
    auto img = this->LoadImage(filename.c_str());
    auto fmt = this->format_registry_.GetImageFormat(&img);
    ASSERT_EQ(expected_format, fmt->Name());
    auto image_info = fmt->Parser()->GetInfo(&img);
    ASSERT_EQ(expected_sh, image_info.shape);
  }

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

TEST_F(ImageFormatTest, DISABLED_Tiff) {
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


}  // namespace test
}  // namespace imgcodec
}  // namespace dali
