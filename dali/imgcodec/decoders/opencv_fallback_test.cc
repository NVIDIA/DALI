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
#include "dali/imgcodec/decoders/decoder_test_helper.h"
#include "dali/imgcodec/decoders/opencv_fallback.h"
#include "dali/imgcodec/parsers/bmp.h"
#include "dali/imgcodec/parsers/tiff.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/core/mm/memory.h"
#include "dali/core/convert.h"

namespace dali {
namespace imgcodec {
namespace test {

namespace {

const auto &dali_extra = dali::testing::dali_extra_path();

auto tiff_dir = dali_extra + "/db/single/tiff/0/";
auto ref_tiff_dir = dali_extra + "/db/single/reference/tiff/0/";

auto tiff_rgb_path = tiff_dir + "/cat-111793_640.tiff";
auto tiff_rgb_ref_path = ref_tiff_dir + "/cat-111793_640.tiff.npy";

auto tiff_gray_path = tiff_dir + "/cat-111793_640_gray.tiff";
auto tiff_gray_ref_path = ref_tiff_dir + "/cat-111793_640_gray.tiff.npy";

}  // namespace



template <typename OutType>
class OpenCVFallbackDecodeTest : public NumpyDecoderTestBase<CPUBackend, OutType> {
 protected:
  std::shared_ptr<ImageDecoderInstance> CreateDecoder() override {
    OpenCVDecoderFactory factory;
    return factory.Create(CPU_ONLY_DEVICE_ID);
  }
  std::shared_ptr<ImageParser> CreateParser() override {
    return std::make_shared<TiffParser>();
  }
  static const auto dtype = type2id<OutType>::value;
};


using OpenCVFallbackDecoderTypes = ::testing::Types<uint8_t, uint16_t, uint32_t>;
TYPED_TEST_SUITE(OpenCVFallbackDecodeTest, OpenCVFallbackDecoderTypes);

TYPED_TEST(OpenCVFallbackDecodeTest, FromFilename) {
  auto ref = this->ReadReferenceFrom(tiff_rgb_ref_path);
  auto src = ImageSource::FromFilename(tiff_rgb_path);
  auto img = this->Decode(&src, {this->dtype});
  this->AssertEqualSatNorm(img, ref);
}

TYPED_TEST(OpenCVFallbackDecodeTest, FromHostMem) {
  auto ref = this->ReadReferenceFrom(tiff_rgb_ref_path);
  auto stream = FileStream::Open(tiff_rgb_path, false, false);
  std::vector<uint8_t> data(stream->Size());
  stream->ReadBytes(data.data(), data.size());
  auto src = ImageSource::FromHostMem(data.data(), data.size());
  auto img = this->Decode(&src, {this->dtype});
  this->AssertEqualSatNorm(img, ref);
}

TYPED_TEST(OpenCVFallbackDecodeTest, ROI) {
  auto ref = this->ReadReferenceFrom(tiff_rgb_ref_path);
  auto src = ImageSource::FromFilename(tiff_rgb_path);
  auto info = this->Parser()->GetInfo(&src);

  ROI roi = {{13, 17}, {info.shape[0] - 55, info.shape[1] - 10}};
  auto img = this->Decode(&src, {this->dtype}, roi);
  this->AssertEqualSatNorm(img, this->Crop(ref, roi));
}

TYPED_TEST(OpenCVFallbackDecodeTest, Gray) {
  auto ref = this->ReadReferenceFrom(tiff_gray_ref_path);
  auto src = ImageSource::FromFilename(tiff_gray_path);
  auto img = this->Decode(&src, {this->dtype, DALI_GRAY});
  this->AssertEqualSatNorm(img, ref);
}

TYPED_TEST(OpenCVFallbackDecodeTest, GrayToRgb) {
  auto ref = this->ReadReferenceFrom(tiff_gray_ref_path);
  auto src = ImageSource::FromFilename(tiff_gray_path);
  auto img = this->Decode(&src, {this->dtype, DALI_RGB});

  EXPECT_EQ(img.shape, TensorShape<-1>({ref.shape()[0], ref.shape()[1], 3}));

  auto r = this->Crop(img.template to_static<3>(), {{0, 0, 0}, {img.shape[0], img.shape[1], 1}});
  auto g = this->Crop(img.template to_static<3>(), {{0, 0, 1}, {img.shape[0], img.shape[1], 2}});
  auto b = this->Crop(img.template to_static<3>(), {{0, 0, 2}, {img.shape[0], img.shape[1], 3}});

  this->AssertEqualSatNorm(r, ref);
  this->AssertEqualSatNorm(g, ref);
  this->AssertEqualSatNorm(b, ref);
}

TYPED_TEST(OpenCVFallbackDecodeTest, BatchedAPI) {
  auto ref = this->ReadReferenceFrom(tiff_rgb_ref_path);

  auto decode_batched = [&](std::vector<std::string> filenames) {
    std::vector<ImageSource> src_objs;
    std::vector<ImageSource*> srcs;
    for (const auto& filename : filenames) {
      src_objs.push_back(ImageSource::FromFilename(filename));
    }
    for (auto &obj : src_objs) {
      srcs.push_back(&obj);
    }
    return this->Decode(make_span(srcs), {this->dtype});
  };

  auto img = decode_batched({tiff_rgb_path, tiff_rgb_path});
  this->AssertEqualSatNorm(img[0], ref);
  this->AssertEqualSatNorm(img[1], ref);
}

TEST(OpenCVFallbackTest, Factory) {
  OpenCVDecoderFactory factory;
  EXPECT_TRUE(factory.IsSupported(CPU_ONLY_DEVICE_ID));
  auto props = factory.GetProperties();
  EXPECT_TRUE(!!(props.supported_input_kinds & InputKind::HostMemory));
  EXPECT_TRUE(!!(props.supported_input_kinds & InputKind::Filename));
  EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::DeviceMemory));
  EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::Stream));

  auto decoder = factory.Create(CPU_ONLY_DEVICE_ID);
  EXPECT_NE(decoder, nullptr);
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
