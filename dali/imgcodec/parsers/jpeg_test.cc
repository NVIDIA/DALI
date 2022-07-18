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

  bool CanParse(std::vector<uint8_t> data) {
    auto src = ImageSource::FromHostMem(data.data(), data.size());
    return parser_.CanParse(&src);
  }

  ImageInfo GetInfo(std::vector<uint8_t> data) {
    auto src = ImageSource::FromHostMem(data.data(), data.size());
    return parser_.GetInfo(&src);
  }

  std::vector<uint8_t> replace(const std::vector<uint8_t> &data, const std::vector<uint8_t> &old_value,
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

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
