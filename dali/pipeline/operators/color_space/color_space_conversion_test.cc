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

template <typename InputImgType>
class ColorSpaceConversionToBGRTest : public GenericConversionTest<InputImgType, BGR> {};

template <typename InputImgType>
class ColorSpaceConversionToRGBTest : public GenericConversionTest<InputImgType, RGB> {};

template <typename InputImgType>
class ColorSpaceConversionToGrayTest : public GenericConversionTest<InputImgType, Gray> {};

template <typename InputImgType>
class ColorSpaceConversionToYCbCrTest : public GenericConversionTest<InputImgType, YCbCr> {};

typedef ::testing::Types<RGB, Gray, YCbCr> ConvertibleToBGR;
TYPED_TEST_SUITE(ColorSpaceConversionToBGRTest, ConvertibleToBGR);

TYPED_TEST(ColorSpaceConversionToBGRTest, test) {
  this->RunTest("ColorSpaceConversion");
}

typedef ::testing::Types<BGR, Gray, YCbCr> ConvertibleToRGB;
TYPED_TEST_SUITE(ColorSpaceConversionToRGBTest, ConvertibleToRGB);

TYPED_TEST(ColorSpaceConversionToRGBTest, test) {
  this->RunTest("ColorSpaceConversion");
}

typedef ::testing::Types<RGB, BGR, YCbCr> ConvertibleToGray;
TYPED_TEST_SUITE(ColorSpaceConversionToGrayTest, ConvertibleToGray);

TYPED_TEST(ColorSpaceConversionToGrayTest, test) {
  this->RunTest("ColorSpaceConversion");
}

typedef ::testing::Types<RGB, BGR, Gray> ConvertibleToYCbCr;
TYPED_TEST_SUITE(ColorSpaceConversionToYCbCrTest, ConvertibleToYCbCr);

TYPED_TEST(ColorSpaceConversionToYCbCrTest, test) {
  this->RunTest("ColorSpaceConversion");
}

}  // namespace dali
