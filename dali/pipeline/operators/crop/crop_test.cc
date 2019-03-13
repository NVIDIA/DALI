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

#include "dali/test/dali_test_matching.h"

namespace dali {

template <typename ImgType>
class CropTest : public GenericMatchingTest<ImgType> {};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_SUITE(CropTest, Types);

const bool addImageType = true;

TYPED_TEST(CropTest, CropVector) {
  this->RunTest({"Crop", {"crop", "224, 256", DALI_FLOAT_VEC}}, addImageType);
}

TYPED_TEST(CropTest, CropSingleDim) {
  this->RunTest({"Crop", {"crop", "224", DALI_FLOAT_VEC}}, addImageType);
}

TYPED_TEST(CropTest, CropWH) {
  const OpArg params[] = {{"crop_h", "224", DALI_FLOAT},
                          {"crop_w", "256", DALI_FLOAT}};
  this->RunTest("Crop", params, 2, addImageType);
}

TYPED_TEST(CropTest, ErrorTooBigWindow1) {
  const OpArg params[] = {{"crop_h", "2240", DALI_FLOAT},
                          {"crop_w", "256", DALI_FLOAT}};
  EXPECT_ANY_THROW(
    this->RunTest("Crop", params, 2, addImageType));
}

TYPED_TEST(CropTest, ErrorTooBigWindow2) {
  const OpArg params[] = {{"crop_h", "224", DALI_FLOAT},
                          {"crop_w", "2560", DALI_FLOAT}};
  EXPECT_ANY_THROW(
    this->RunTest("Crop", params, 2, addImageType));
}

TYPED_TEST(CropTest, ErrorTooBigWindow3) {
  const OpArg params[] = {{"crop", "2240", DALI_FLOAT_VEC}};
  EXPECT_ANY_THROW(
    this->RunTest("Crop", params, 1, addImageType));
}

TYPED_TEST(CropTest, ErrorTooBigWindow4) {
  const OpArg params[] = {{"crop", "2240, 256", DALI_FLOAT_VEC}};
  EXPECT_ANY_THROW(
    this->RunTest("Crop", params, 1, addImageType));
}

TYPED_TEST(CropTest, ErrorTooBigWindow5) {
  const OpArg params[] = {{"crop", "224, 2560", DALI_FLOAT_VEC}};
  EXPECT_ANY_THROW(
    this->RunTest("Crop", params, 1, addImageType));
}

TYPED_TEST(CropTest, ErrorWrongArgs_NoCropWindowDims) {
  EXPECT_ANY_THROW(
    this->RunTest("Crop", nullptr, 0, addImageType));
}

TYPED_TEST(CropTest, ErrorWrongArgs2_BothCropAndCropHW) {
  const OpArg params[] = {{"crop", "224, 256", DALI_FLOAT_VEC},
                          {"crop_h", "224", DALI_FLOAT},
                          {"crop_w", "256", DALI_FLOAT}};
  EXPECT_ANY_THROW(
    this->RunTest("Crop", params, 3, addImageType));
}

TYPED_TEST(CropTest, ErrorWrongArgs3_BothCropAndCropH) {
  const OpArg params[] = {{"crop", "224, 256", DALI_FLOAT_VEC},
                          {"crop_h", "224", DALI_FLOAT}};
  EXPECT_ANY_THROW(
    this->RunTest("Crop", params, 2, addImageType));
}

TYPED_TEST(CropTest, ErrorWrongArgs_OnlyH) {
  const OpArg params[] = {{"crop_h", "224", DALI_FLOAT}};
  EXPECT_ANY_THROW(
    this->RunTest("Crop", params, 1, addImageType));
}


}  // namespace dali
