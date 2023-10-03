// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

const auto img_dir = join(dali::testing::dali_extra_path(), "db/single/jpeg");
const auto ref_dir = join(dali::testing::dali_extra_path(), "db/single/reference/jpeg");
const auto jpeg_image = join(img_dir, "134/site-1534685_1280.jpg");
const auto ref_prefix = join(ref_dir, "site-1534685_1280");

const auto jpeg_image1 = join(img_dir, "113/snail-4291306_1280.jpg");
const auto ref_prefix1 = join(ref_dir, "snail-4291306_1280");

const auto jpeg_image2 = join(img_dir, "100/swan-3584559_640.jpg");
const auto ref_prefix2 = join(ref_dir, "swan-3584559_640");

}  // namespace

TEST(LibJpegTurboDecoderTest, Factory) {
  LibJpegTurboDecoderFactory factory;
  EXPECT_TRUE(factory.IsSupported(CPU_ONLY_DEVICE_ID));
  auto props = factory.GetProperties();
  EXPECT_TRUE(!!(props.supported_input_kinds & InputKind::HostMemory));;
  EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::Filename));;
  EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::DeviceMemory));;
  EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::Stream));

  std::map<string, std::any> params = { { "fast_idct", false } };
  auto decoder = factory.Create(CPU_ONLY_DEVICE_ID, params);
  EXPECT_NE(decoder, nullptr);
  EXPECT_EQ(std::any_cast<bool>(decoder->GetParam("fast_idct")), false);

  decoder.reset();
  params = { { "fast_idct", true } };
  decoder = factory.Create(CPU_ONLY_DEVICE_ID, params);
  EXPECT_NE(decoder, nullptr);
  EXPECT_EQ(std::any_cast<bool>(decoder->GetParam("fast_idct")), true);
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

  DecodeParams GetParams(DALIImageType color_fmt) {
    DecodeParams opts{};
    opts.dtype = dtype;
    opts.format = color_fmt;
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

  std::string GetPath(const std::string &prefix, DALIImageType color_fmt) {
    if (color_fmt == DALI_YCbCr) {
      return make_string(prefix, "_ycbcr.npy");
    } else if (color_fmt == DALI_GRAY) {
      return make_string(prefix, "_gray.npy");
    } else {
      assert(color_fmt == DALI_RGB);
      return make_string(prefix, ".npy");
    }
  }

  void TestDecodeSingleAPI(DALIImageType color_fmt) {
    ImageBuffer image(jpeg_image);
    auto decoded = this->Decode(&image.src, this->GetParams(color_fmt));
    auto ref = this->ReadReferenceFrom(GetPath(ref_prefix, color_fmt));
    if (color_fmt != DALI_RGB) {
      AssertClose(decoded, ref, this->GetEps());
    } else {
      AssertEqualSatNorm(decoded, ref);
    }
  }

  void TestDecodeBatchAPI(DALIImageType color_fmt) {
    auto ref0 = this->ReadReferenceFrom(GetPath(ref_prefix, color_fmt));
    auto ref1 = this->ReadReferenceFrom(GetPath(ref_prefix1, color_fmt));
    auto ref2 = this->ReadReferenceFrom(GetPath(ref_prefix2, color_fmt));
    ImageBuffer image0(jpeg_image);
    ImageBuffer image1(jpeg_image1);
    ImageBuffer image2(jpeg_image2);

    std::vector<ImageSource*> srcs = {&image0.src, &image1.src, &image2.src};
    auto img = this->Decode(make_span(srcs), this->GetParams(color_fmt));

    if (color_fmt != DALI_RGB) {
      auto eps = this->GetEps();
      AssertClose(img[0], ref0, eps);
      AssertClose(img[1], ref1, eps);
      AssertClose(img[2], ref2, eps);
    } else {
      AssertEqualSatNorm(img[0], ref0);
      AssertEqualSatNorm(img[1], ref1);
      AssertEqualSatNorm(img[2], ref2);
    }
  }
};

using DecodeOutputTypes = ::testing::Types<uint8_t, int16_t, float>;
TYPED_TEST_SUITE(LibJpegTurboDecoderTest, DecodeOutputTypes);

TYPED_TEST(LibJpegTurboDecoderTest, Decode) {
  this->TestDecodeSingleAPI(DALI_RGB);
}

TYPED_TEST(LibJpegTurboDecoderTest, DecodeBatch) {
  this->TestDecodeBatchAPI(DALI_RGB);
}

TYPED_TEST(LibJpegTurboDecoderTest, DecodeRoi) {
  ImageBuffer image(jpeg_image);
  auto decoded = this->Decode(&image.src, this->GetParams(DALI_RGB), {{5, 20}, {800, 1000}});
  auto ref = this->ReadReferenceFrom(make_string(ref_prefix, "_roi.npy"));
  AssertEqualSatNorm(decoded, ref);
}

TYPED_TEST(LibJpegTurboDecoderTest, DecodeYCbCr) {
  this->TestDecodeSingleAPI(DALI_YCbCr);
}

TYPED_TEST(LibJpegTurboDecoderTest, DecodeBatchYCbCr) {
  this->TestDecodeBatchAPI(DALI_YCbCr);
}

TYPED_TEST(LibJpegTurboDecoderTest, DecodeGray) {
  this->TestDecodeSingleAPI(DALI_GRAY);
}

TYPED_TEST(LibJpegTurboDecoderTest, DecodeBatchGray) {
  this->TestDecodeBatchAPI(DALI_GRAY);
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
