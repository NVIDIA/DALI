// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/test/dali_test_conversion.h"

namespace dali {

// TODO(janton) : enable tests with 4-channel as input type (need to support it in host decoder)
typedef ::testing::Types<RGB, BGR, Gray, YCbCr/*, RGBA, BGRA, ARGB, ABGR*/> AllColorSpaces;

template <typename InputImgType>
class ColorSpaceConversionToRGBTest : public GenericConversionTest<InputImgType, RGB> {};

template <typename InputImgType>
class ColorSpaceConversionToBGRTest : public GenericConversionTest<InputImgType, BGR> {};

template <typename InputImgType>
class ColorSpaceConversionToGrayTest : public GenericConversionTest<InputImgType, Gray> {};

template <typename InputImgType>
class ColorSpaceConversionToYCbCrTest : public GenericConversionTest<InputImgType, YCbCr> {};

template <typename InputImgType>
class ColorSpaceConversionToRGBATest : public GenericConversionTest<InputImgType, RGBA> {};

template <typename InputImgType>
class ColorSpaceConversionToBGRATest : public GenericConversionTest<InputImgType, BGRA> {};

template <typename InputImgType>
class ColorSpaceConversionToARGBTest : public GenericConversionTest<InputImgType, ARGB> {};

template <typename InputImgType>
class ColorSpaceConversionToABGRTest : public GenericConversionTest<InputImgType, ABGR> {};

TYPED_TEST_CASE(ColorSpaceConversionToBGRTest, AllColorSpaces);
TYPED_TEST(ColorSpaceConversionToBGRTest, test) {
  this->RunTest("ColorSpaceConversion");
}

TYPED_TEST_CASE(ColorSpaceConversionToRGBTest, AllColorSpaces);
TYPED_TEST(ColorSpaceConversionToRGBTest, test) {
  this->RunTest("ColorSpaceConversion");
}

TYPED_TEST_CASE(ColorSpaceConversionToGrayTest, AllColorSpaces);
TYPED_TEST(ColorSpaceConversionToGrayTest, test) {
  this->RunTest("ColorSpaceConversion");
}

TYPED_TEST_CASE(ColorSpaceConversionToYCbCrTest, AllColorSpaces);
TYPED_TEST(ColorSpaceConversionToYCbCrTest, test) {
  this->RunTest("ColorSpaceConversion");
}

TYPED_TEST_CASE(ColorSpaceConversionToRGBATest, AllColorSpaces);
TYPED_TEST(ColorSpaceConversionToRGBATest, test) {
  this->RunTest("ColorSpaceConversion");
}

TYPED_TEST_CASE(ColorSpaceConversionToARGBTest, AllColorSpaces);
TYPED_TEST(ColorSpaceConversionToARGBTest, test) {
  this->RunTest("ColorSpaceConversion");
}

TYPED_TEST_CASE(ColorSpaceConversionToBGRATest, AllColorSpaces);
TYPED_TEST(ColorSpaceConversionToBGRATest, test) {
  this->RunTest("ColorSpaceConversion");
}

TYPED_TEST_CASE(ColorSpaceConversionToABGRTest, AllColorSpaces);
TYPED_TEST(ColorSpaceConversionToABGRTest, test) {
  this->RunTest("ColorSpaceConversion");
}

}  // namespace dali
