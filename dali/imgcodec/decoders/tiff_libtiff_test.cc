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

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "dali/imgcodec/decoders/tiff_libtiff.h"
#include "dali/imgcodec/parsers/tiff.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/core/mm/memory.h"
#include "dali/core/convert.h"
#include "dali/imgcodec/decoders/decoder_test_helper.h"

namespace dali {
namespace imgcodec {
namespace test {

namespace {
const auto &dali_extra = dali::testing::dali_extra_path();
auto img_dir = dali_extra + "/db/single/tiff/0/";
auto ref_dir = dali_extra + "/db/single/reference/tiff/0/";

auto rgb_path = img_dir + "/cat-111793_640.tiff";
auto rgb_ref_path = ref_dir + "/cat-111793_640.tiff.npy";

auto gray_path = img_dir + "/cat-111793_640_bw.tiff";
auto gray_ref_path = ref_dir + "/cat-111793_640_bw.tiff.npy";

auto palette_path = img_dir + "/cat-300572_640_palette.tiff";
}  // namespace

class LibTiffDecoderTest : public NumpyDecoderTestBase<uint8_t> {
 protected:
  std::shared_ptr<ImageDecoderInstance> CreateDecoder(ThreadPool &tp) override {
    LibTiffDecoder decoder;
    return decoder.Create(CPU_ONLY_DEVICE_ID, tp);
  }
  std::shared_ptr<ImageParser> CreateParser() override {
    return std::make_shared<TiffParser>();
  }
};

TEST_F(LibTiffDecoderTest, Test) {
  auto ref = ReadReferenceFrom(rgb_ref_path);
  auto src = ImageSource::FromFilename(rgb_path);
  auto img = Decode(&src);
  AssertEqualSatNorm(img, ref);
}

TEST_F(LibTiffDecoderTest, TestROI) {
  auto ref = ReadReferenceFrom(rgb_ref_path);
  auto src = ImageSource::FromFilename(rgb_path);
  auto info = Parser()->GetInfo(&src);

  DecodeParams params = {};
  ROI roi = {{13, 17, 0}, {info.shape[0] - 55, info.shape[1] - 10, 3}};
  auto img = Decode(&src, params, roi);
  AssertEqualSatNorm(img, Crop(ref, roi));
}

TEST_F(LibTiffDecoderTest, TestRgbToGray) {
  auto ref = ReadReferenceFrom(gray_ref_path);
  auto src = ImageSource::FromFilename(rgb_path);
  auto img = Decode(&src, {.format = DALI_GRAY});
  AssertEqualSatNorm(img, ref);
}

TEST_F(LibTiffDecoderTest, TestGray) {
  auto ref = ReadReferenceFrom(gray_ref_path);
  auto src = ImageSource::FromFilename(gray_path);
  auto img = Decode(&src, {.format = DALI_GRAY});
  AssertEqualSatNorm(img, ref);
}

TEST_F(LibTiffDecoderTest, TestGrayToRgb) {
  auto ref = ReadReferenceFrom(gray_ref_path);
  auto src = ImageSource::FromFilename(gray_path);
  auto img = Decode(&src, {.format = DALI_RGB});

  EXPECT_EQ(img.shape(), TensorShape<-1>({ref.shape()[0], ref.shape()[1], 3}));

  AssertEqualSatNorm(Crop(img, {{0, 0, 0}, {img.shape()[0], img.shape()[1], 1}}), ref);
  AssertEqualSatNorm(Crop(img, {{0, 0, 1}, {img.shape()[0], img.shape()[1], 2}}), ref);
  AssertEqualSatNorm(Crop(img, {{0, 0, 2}, {img.shape()[0], img.shape()[1], 3}}), ref);
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
