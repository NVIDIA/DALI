// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/test/dali_test_decoder.h"

namespace dali {

template <typename ImgType>
class HostDecodeTest : public GenericDecoderTest<ImgType> {
 protected:
  uint32_t GetImageLoadingFlags() const override {
    return t_loadJPEGs + t_loadPNGs;
  }

  const OpSpec DecodingOp() const override {
    return OpSpec("HostDecoder")
      .AddArg("device", "cpu")
      .AddArg("output_type", this->img_type_)
      .AddInput("encoded", "cpu")
      .AddOutput("decoded", "cpu");
  }

  uint32_t GetTestCheckType() const  override {
    return t_checkColorComp + t_checkElements;  // + t_checkAll + t_checkNoAssert;
  }
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(HostDecodeTest, Types);

typedef ::testing::Types<RGB, BGR, Gray> Types;


template<typename ImageType>
class HostDecodeTestJpeg : public HostDecodeTest<ImageType> {
 protected:
  uint32_t GetImageLoadingFlags() const override {
    return t_jpegImgType;
  }
};

TYPED_TEST_CASE(HostDecodeTestJpeg, Types);

TYPED_TEST(HostDecodeTestJpeg, TestJpegDecode) {
  this->RunTestDecode(t_jpegImgType, 0.65);
}


template<typename ImageType>
class HostDecodeTestPng : public HostDecodeTest<ImageType> {
 protected:
  uint32_t GetImageLoadingFlags() const override {
    return t_pngImgType;
  }
};

TYPED_TEST_CASE(HostDecodeTestPng, Types);

TYPED_TEST(HostDecodeTestPng, TestPngDecode) {
  this->RunTestDecode(t_pngImgType);
}


template<typename ImageType>
class HostDecodeTestTiff : public HostDecodeTest<ImageType> {
 protected:
  uint32_t GetImageLoadingFlags() const override {
    return t_tiffImgType;
  }
};

TYPED_TEST_CASE(HostDecodeTestTiff, Types);

TYPED_TEST(HostDecodeTestTiff, TestTiffDecode) {
  this->RunTestDecode(t_tiffImgType);
}
}  // namespace dali
