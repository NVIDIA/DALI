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
    image_parser_mgr_.RegisterParser(std::make_shared<JpegParser>());
    image_parser_mgr_.RegisterParser(std::make_shared<PngParser>());
    image_parser_mgr_.RegisterParser(std::make_shared<BmpParser>());
    image_parser_mgr_.RegisterParser(std::make_shared<TiffParser>());
    image_parser_mgr_.RegisterParser(std::make_shared<PnmParser>());
    image_parser_mgr_.RegisterParser(std::make_shared<Jpeg2000Parser>());
    image_parser_mgr_.RegisterParser(std::make_shared<WebpParser>());
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

  ImageParserManager image_parser_mgr_;
  std::vector<char> data_;
};

TEST_F(ImageFormatTest, DISABLED_Jpeg) {
  auto filename = testing::dali_extra_path() + "/db/single/jpeg/372/baboon-174073_1280.jpg";
  auto img = this->LoadImage(filename.c_str());
  auto image_info = this->image_parser_mgr_.Parse(&img);
  ASSERT_EQ(TensorShape<>(720, 1280, 3), image_info.shape);
}

TEST_F(ImageFormatTest, DISABLED_Png) {
  auto filename = testing::dali_extra_path() + "/db/single/png/0/cat-3504008_640.png";
  auto img = this->LoadImage(filename.c_str());
  auto image_info = this->image_parser_mgr_.Parse(&img);
  ASSERT_EQ(TensorShape<>(425, 640, 3), image_info.shape);
}

TEST_F(ImageFormatTest, Bmp) {
  auto filename = testing::dali_extra_path() + "/db/single/bmp/0/cat-1046544_640.bmp";
  auto img = this->LoadImage(filename.c_str());
  auto image_info = this->image_parser_mgr_.Parse(&img);
  ASSERT_EQ(TensorShape<>(475, 640, 3), image_info.shape);
}

TEST_F(ImageFormatTest, DISABLED_Tiff) {
  auto filename = testing::dali_extra_path() + "/db/single/tiff/0/cat-1245673_640.tiff";
  auto img = this->LoadImage(filename.c_str());
  auto image_info = this->image_parser_mgr_.Parse(&img);
  ASSERT_EQ(TensorShape<>(423, 640, 3), image_info.shape);
}

TEST_F(ImageFormatTest, DISABLED_Pnm) {
  auto filename = testing::dali_extra_path() + "/db/single/pnm/0/cat-300572_640.pnm";
  auto img = this->LoadImage(filename.c_str());
  auto image_info = this->image_parser_mgr_.Parse(&img);
  ASSERT_EQ(TensorShape<>(536, 640, 3), image_info.shape);
}

TEST_F(ImageFormatTest, DISABLED_Jpeg2000) {
  auto filename = testing::dali_extra_path() + "/db/single/jpeg2k/0/cat-3113513_640.jp2";
  auto img = this->LoadImage(filename.c_str());
  auto image_info = this->image_parser_mgr_.Parse(&img);
  ASSERT_EQ(TensorShape<>(299, 640, 3), image_info.shape);
}

TEST_F(ImageFormatTest, DISABLED_Webp) {
  auto filename = testing::dali_extra_path() + "/db/single/webp/lossy/cat-1245673_640.webp";
  auto img = this->LoadImage(filename.c_str());
  auto image_info = this->image_parser_mgr_.Parse(&img);
  ASSERT_EQ(TensorShape<>(423, 640, 3), image_info.shape);
}


}  // namespace test
}  // namespace imgcodec
}  // namespace dali
