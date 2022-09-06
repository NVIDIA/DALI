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
#include "dali/imgcodec/decoders/libjpeg_turbo.h"
#include "dali/imgcodec/decoders/decoder_test_helper.h"
#include "dali/imgcodec/parsers/jpeg.h"
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
template<typename... Args>
std::string join(Args... args) {
  return make_string_delim('/', args...);
}

std::vector<uint8_t> read_file(const std::string &filename) {
    std::ifstream stream(filename, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

struct ImageBuffer {
  std::vector<uint8_t> buffer;
  ImageSource src;
  explicit ImageBuffer(const std::string &filename)
  : buffer(read_file(filename))
  , src(ImageSource::FromHostMem(buffer.data(), buffer.size())) {}
};

const auto img_dir = join(dali::testing::dali_extra_path(), "db/single/jpeg");
const auto ref_dir = join(dali::testing::dali_extra_path(), "db/single/reference/jpeg");
const auto jpeg_image = join(img_dir, "134/site-1534685_1280.jpg");
const auto ref_prefix = join(ref_dir, "site-1534685_1280");
}  // namespace

TEST(LibJpegTurboDecoderTest, Factory) {
  LibJpegTurboDecoderFactory factory;
  EXPECT_TRUE(factory.IsSupported(CPU_ONLY_DEVICE_ID));
  auto props = factory.GetProperties();
  EXPECT_TRUE(!!(props.supported_input_kinds & InputKind::HostMemory));;
  EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::Filename));;
  EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::DeviceMemory));;
  EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::Stream));

  std::map<string, any> params = { { "fast_idct", false } };
  auto decoder = factory.Create(CPU_ONLY_DEVICE_ID, params);
  EXPECT_NE(decoder, nullptr);
  EXPECT_EQ(any_cast<bool>(decoder->GetParam("fast_idct")), false);

  decoder.reset();
  params = { { "fast_idct", true } };
  decoder = factory.Create(CPU_ONLY_DEVICE_ID, params);
  EXPECT_NE(decoder, nullptr);
  EXPECT_EQ(any_cast<bool>(decoder->GetParam("fast_idct")), true);
}

template<typename OutputType>
class LibJpegTurboDecoderTest : public NumpyDecoderTestBase<CPUBackend, OutputType> {
 protected:
  static const auto dtype = type2id<OutputType>::value;

  std::shared_ptr<ImageDecoderInstance> CreateDecoder() override {
    return LibJpegTurboDecoderFactory().Create(CPU_ONLY_DEVICE_ID);
  }

  std::shared_ptr<ImageParser> CreateParser() override {
    return std::make_shared<JpegParser>();
  }

  DecodeParams GetParams() {
    DecodeParams opts{};
    opts.dtype = dtype;
    return opts;
  }

  float GetEps() {
    if (std::is_floating_point_v<OutputType>) {
      return 0.01f;
    } else {
      // Adjusting the epsilon to OutputType
      return 0.01 * max_value<OutputType>();
    }
  }
};

using DecodeOutputTypes = ::testing::Types<uint8_t, int16_t, float>;
TYPED_TEST_SUITE(LibJpegTurboDecoderTest, DecodeOutputTypes);

TYPED_TEST(LibJpegTurboDecoderTest, Decode) {
  ImageBuffer image(jpeg_image);
  auto decoded = this->Decode(&image.src, this->GetParams());
  auto ref = this->ReadReferenceFrom(make_string(ref_prefix, ".npy"));
  this->AssertEqualSatNorm(decoded, ref);
}

TYPED_TEST(LibJpegTurboDecoderTest, DecodeRoi) {
  ImageBuffer image(jpeg_image);
  auto decoded = this->Decode(&image.src, this->GetParams(), {{5, 20}, {800, 1000}});
  auto ref = this->ReadReferenceFrom(make_string(ref_prefix, "_roi.npy"));
  this->AssertEqualSatNorm(decoded, ref);
}

TYPED_TEST(LibJpegTurboDecoderTest, DecodeYCbCr) {
  ImageBuffer image(jpeg_image);
  auto params = this->GetParams();
  params.format = DALI_YCbCr;
  auto decoded = this->Decode(&image.src, params);
  auto ref = this->ReadReferenceFrom(make_string(ref_prefix, "_ycbcr.npy"));
  this->AssertClose(decoded, ref, this->GetDelta());
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
