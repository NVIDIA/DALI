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

#include "dali/test/dali_test_bboxes.h"

namespace dali {

template <typename ImgType>
class RandomBBoxCropTest : public GenericBBoxesTest<ImgType> {};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_SUITE(RandomBBoxCropTest, Types);

const bool addImageType = true;

TYPED_TEST(RandomBBoxCropTest, CreateWithSingleThreshold) {
  this->RunBBoxesCPU({"RandomBBoxCrop", {"thresholds", "0.0", DALI_FLOAT_VEC}},
                     addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithMultipleThresholds) {
  this->RunBBoxesCPU(
      {"RandomBBoxCrop", {"thresholds", "0.0, 0.1, 1.0", DALI_FLOAT_VEC}},
      addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithoutThreshold) {
  EXPECT_THROW(
      this->RunBBoxesCPU({"RandomBBoxCrop", {"thresholds", "", DALI_FLOAT_VEC}},
                         addImageType),
      std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithThresholdTooLargeFails) {
  EXPECT_THROW(this->RunBBoxesCPU({"RandomBBoxCrop",
                                   {"thresholds", "0.0, 1.1", DALI_FLOAT_VEC}},
                                  addImageType),
               std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithThresholdTooSmallFails) {
  EXPECT_THROW(this->RunBBoxesCPU({"RandomBBoxCrop",
                                   {"thresholds", "-0.1, 1.0", DALI_FLOAT_VEC}},
                                  addImageType),
               std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithThresholdsProvidedInDecreasingOrder) {
  this->RunBBoxesCPU(
      {"RandomBBoxCrop", {"thresholds", "1,0, 0.5, 0.0", DALI_FLOAT_VEC}},
      addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithThresholdsRepeated) {
  this->RunBBoxesCPU(
      {"RandomBBoxCrop", {"thresholds", "1,0, 1.0", DALI_FLOAT_VEC}},
      addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithAspectRatio) {
  this->RunBBoxesCPU(
      {"RandomBBoxCrop", {"aspect_ratio", "0.0, 1.0", DALI_FLOAT_VEC}},
      addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithSingleAspectRatio) {
  EXPECT_THROW(this->RunBBoxesCPU(
                   {"RandomBBoxCrop", {"aspect_ratio", "0.0", DALI_FLOAT_VEC}},
                   addImageType),
               std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithEmptyAspectRatio) {
  EXPECT_THROW(this->RunBBoxesCPU(
                   {"RandomBBoxCrop", {"aspect_ratio", "", DALI_FLOAT_VEC}},
                   addImageType),
               std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithNegativeAspectRatio) {
  EXPECT_THROW(
      this->RunBBoxesCPU(
          {"RandomBBoxCrop", {"aspect_ratio", "-0.1, 1.0", DALI_FLOAT_VEC}},
          addImageType),
      std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithVeryLargeAspectRatio) {
  this->RunBBoxesCPU(
      {"RandomBBoxCrop", {"aspect_ratio", "0.0, 666.0", DALI_FLOAT_VEC}},
      addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithSwappedOrderAspectRatio) {
  EXPECT_THROW(
      this->RunBBoxesCPU(
          {"RandomBBoxCrop", {"aspect_ratio", "1.0, 0.0", DALI_FLOAT_VEC}},
          addImageType),
      std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithScaling) {
  this->RunBBoxesCPU(
      {"RandomBBoxCrop", {"scaling", "0.0, 1.0", DALI_FLOAT_VEC}},
      addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithSingleScaling) {
  EXPECT_THROW(
      this->RunBBoxesCPU({"RandomBBoxCrop", {"scaling", "0.0", DALI_FLOAT_VEC}},
                         addImageType),
      std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithEmptyScaling) {
  EXPECT_THROW(
      this->RunBBoxesCPU({"RandomBBoxCrop", {"scaling", "", DALI_FLOAT_VEC}},
                         addImageType),
      std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithNegativeScaling) {
  EXPECT_THROW(this->RunBBoxesCPU(
                   {"RandomBBoxCrop", {"scaling", "-0.1, 1.0", DALI_FLOAT_VEC}},
                   addImageType),
               std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithVeryLargeScaling) {
  this->RunBBoxesCPU(
      {"RandomBBoxCrop", {"scaling", "0.0, 666.0", DALI_FLOAT_VEC}},
      addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithSwappedOrderScaling) {
  EXPECT_THROW(this->RunBBoxesCPU(
                   {"RandomBBoxCrop", {"scaling", "1.0, 0.0", DALI_FLOAT_VEC}},
                   addImageType),
               std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithLtrb) {
  this->RunBBoxesCPU({"RandomBBoxCrop", {"ltrb", "true", DALI_BOOL}},
                     addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithXywh) {
  this->RunBBoxesCPU({"RandomBBoxCrop", {"ltrb", "false", DALI_BOOL}},
                     addImageType, false);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithNumAttemptsNegative) {
  EXPECT_THROW(
      this->RunBBoxesCPU({"RandomBBoxCrop", {"num_attempts", "-1", DALI_INT32}},
                         addImageType),
      std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithNumAttemptsZero) {
  EXPECT_THROW(
      this->RunBBoxesCPU({"RandomBBoxCrop", {"num_attempts", "0", DALI_INT32}},
                         addImageType),
      std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithNumAttemptsGreaterThanZero) {
  this->RunBBoxesCPU({"RandomBBoxCrop", {"num_attempts", "1", DALI_INT32}},
                     addImageType);
}

}  // namespace dali
