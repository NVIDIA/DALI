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
#include <experimental/filesystem>
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
#include "dali/image/image_factory.h"

namespace fs = std::experimental::filesystem;

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


// @brief Base class for tests comparing imgcodec with some other implementation
class ComparisonTestBase : public ImageFormatTest {
 protected:
  /// @brief This method should return shape from the other implementation
  virtual TensorShape<> ShapeOf(const std::string &filenames) const = 0;

  void Run(const std::string &filename, std::string expected_format) {
    Test(filename, expected_format, ShapeOf(filename));
  }

 public:
  void RunOnDirectory(std::string directory, std::string expected_format,
                      std::vector<std::string> extensions) {
    unsigned nimages = 0;

    for (const auto &entry : fs::recursive_directory_iterator(directory)) {
      if (fs::is_regular_file(entry.path())) {
        const auto path = entry.path().string();
        for (const auto& ext : extensions) {
          if (path.substr(path.size() - ext.size(), ext.size()) == ext) {
            Run(path, expected_format);
            nimages++;
          }
        }
      }
    }

    if (nimages == 0)
      FAIL() << "No matching images in " << directory;
  }
};

/**
 * @brief Compares imgcodec's parser with the old parsers
 *
 * Compares shapes returned by imgcodec's GetInfo with those returned by PeekShape
 * from the old implementation.
 */
class CompatibilityTest : public ComparisonTestBase {
 protected:
  TensorShape<> ShapeOf(const std::string &filename) const {
    SCOPED_TRACE(filename);
    auto src = ImageSource::FromFilename(filename);
    auto stream = src.Open();
    std::vector<uint8_t> data(stream->Size());
    EXPECT_EQ(data.size(), stream->Read(data.data(), data.size()));
    auto img = ImageFactory::CreateImage(data.data(), data.size(), DALI_RGB);
    auto shape = img->PeekShape();
    return shape;
  }
};

TEST_F(ImageFormatTest, Jpeg) {
  Test(testing::dali_extra_path() + "/db/single/jpeg/372/baboon-174073_1280.jpg", "jpeg",
       TensorShape<>(720, 1280, 3));
}

TEST_F(ImageFormatTest, DISABLED_Png) {
  Test(testing::dali_extra_path() + "/db/single/png/0/cat-3504008_640.png", "png",
       TensorShape<>(425, 640, 3));
}

TEST_F(ImageFormatTest, DISABLED_Bmp) {
  Test(testing::dali_extra_path() + "/db/single/bmp/0/cat-1046544_640.bmp", "bmp",
       TensorShape<>(475, 640, 3));
}

TEST_F(ImageFormatTest, Tiff) {
  Test(testing::dali_extra_path() + "/db/single/tiff/0/cat-1245673_640.tiff", "tiff",
       TensorShape<>(423, 640, 3));
}

TEST_F(ImageFormatTest, Tiff_Palette) {
  Test(testing::dali_extra_path() + "/db/single/tiff/0/cat-300572_640_palette.tiff", "tiff",
       TensorShape<>(536, 640, 3));
}

TEST_F(ImageFormatTest, Pnm) {
  Test(testing::dali_extra_path() + "/db/single/pnm/0/cat-300572_640.pnm", "pnm",
       TensorShape<>(536, 640, 3));
}

TEST_F(ImageFormatTest, Jpeg2000) {
  Test(testing::dali_extra_path() + "/db/single/jpeg2k/0/cat-3113513_640.jp2", "jpeg2000",
       TensorShape<>(299, 640, 3));
}

TEST_F(ImageFormatTest, Webp) {
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

TEST_F(CompatibilityTest, DISABLED_Png) {
  RunOnDirectory(testing::dali_extra_path() + "/db/single/png/", "png", {".png"});
}

TEST_F(CompatibilityTest, DISABLED_Bmp) {
  RunOnDirectory(testing::dali_extra_path() + "/db/single/bmp/", "bmp", {".bmp"});
}

TEST_F(CompatibilityTest, Tiff) {
  RunOnDirectory(testing::dali_extra_path() + "/db/single/tiff/", "tiff", {".tiff"});
}

TEST_F(CompatibilityTest, DISABLED_Pnm) {
  RunOnDirectory(testing::dali_extra_path() + "/db/single/pnm/", "pnm",
                 {".pnm", ".ppm", ".pgm", ".pbm"});
}

TEST_F(CompatibilityTest, DISABLED_Jpeg2000) {
  RunOnDirectory(testing::dali_extra_path() + "/db/single/jpeg2k/", "jpeg2000", {".jp2"});
}

TEST_F(CompatibilityTest, DISABLED_WebP) {
  RunOnDirectory(testing::dali_extra_path() + "/db/single/webp/", "webp", {".webp"});
}

TEST_F(CompatibilityTest, DISABLED_Jpeg) {
  RunOnDirectory(testing::dali_extra_path() + "/db/single/jpeg/", "jpeg", {".jpg", ".jpeg"});
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
