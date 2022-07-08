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

  std::vector<char> valid_tiff_;
  TiffParser parser_;
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

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
