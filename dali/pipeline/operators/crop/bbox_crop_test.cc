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
TYPED_TEST_CASE(RandomBBoxCropTest, Types);

const bool addImageType = true;

TYPED_TEST(RandomBBoxCropTest, CreateWithSingleThreshold) {
  this->RunTest({"RandomBBoxCrop", {"thresholds", "0.0", DALI_FLOAT_VEC}}, addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithMultipleThresholds) {
  this->RunTest({"RandomBBoxCrop", {"thresholds", "0.0, 0.1, 1.0", DALI_FLOAT_VEC}}, addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithoutThreshold) {
  EXPECT_THROW(this->RunTest({"RandomBBoxCrop", {"thresholds", "", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithThresholdTooLargeFails) {
  EXPECT_THROW(this->RunTest({"RandomBBoxCrop", {"thresholds", "0.0, 1.1", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithThresholdTooSmallFails) {
  EXPECT_THROW(this->RunTest({"RandomBBoxCrop", {"thresholds", "-0.1, 1.0", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithThresholdsProvidedInDecreasingOrder) {
  this->RunTest({"RandomBBoxCrop", {"thresholds", "1,0, 0.5, 0.0", DALI_FLOAT_VEC}}, addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithThresholdsRepeated) {
  this->RunTest({"RandomBBoxCrop", {"thresholds", "1,0, 1.0", DALI_FLOAT_VEC}}, addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithAspectRatio) {
  this->RunTest({"RandomBBoxCrop", {"aspect_ratio", "0.0, 1.0", DALI_FLOAT_VEC}}, addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithSingleAspectRatio) {
  EXPECT_THROW(this->RunTest({"RandomBBoxCrop", {"aspect_ratio", "0.0", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithEmptyAspectRatio) {
  EXPECT_THROW(this->RunTest({"RandomBBoxCrop", {"aspect_ratio", "", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithNegativeAspectRatio) {
  EXPECT_THROW(this->RunTest({"RandomBBoxCrop", {"aspect_ratio", "-0.1, 1.0", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithVeryLargeAspectRatio) {
  this->RunTest({"RandomBBoxCrop", {"aspect_ratio", "0.0, 666.0", DALI_FLOAT_VEC}}, addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithSwappedOrderAspectRatio) {
  EXPECT_THROW(this->RunTest({"RandomBBoxCrop", {"aspect_ratio", "1.0, 0.0", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithScaling) {
  this->RunTest({"RandomBBoxCrop", {"scaling", "0.0, 1.0", DALI_FLOAT_VEC}}, addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithSingleScaling) {
  EXPECT_THROW(this->RunTest({"RandomBBoxCrop", {"scaling", "0.0", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithEmptyScaling) {
  EXPECT_THROW(this->RunTest({"RandomBBoxCrop", {"scaling", "", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithNegativeScaling) {
  EXPECT_THROW(this->RunTest({"RandomBBoxCrop", {"scaling", "-0.1, 1.0", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithVeryLargeScaling) {
  this->RunTest({"RandomBBoxCrop", {"scaling", "0.0, 666.0", DALI_FLOAT_VEC}}, addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithSwappedOrderScaling) {
  EXPECT_THROW(this->RunTest({"RandomBBoxCrop", {"scaling", "1.0, 0.0", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithLtrb) {
  this->RunTest({"RandomBBoxCrop", {"ltrb", "true", DALI_BOOL}}, addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithXywh) {
  this->RunTest({"RandomBBoxCrop", {"ltrb", "false", DALI_BOOL}}, addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithNumAttemptsNegative) {
  EXPECT_THROW(this->RunTest({"RandomBBoxCrop", {"num_attempts", "-1", DALI_INT32}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithNumAttemptsZero) {
  EXPECT_THROW(this->RunTest({"RandomBBoxCrop", {"num_attempts", "0", DALI_INT32}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithNumAttemptsGreaterThanZero) {
  this->RunTest({"RandomBBoxCrop", {"num_attempts", "1", DALI_INT32}}, addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithSingleThresholdGPU) {
  this->RunTestGPU({"RandomBBoxCrop", {"thresholds", "0.0", DALI_FLOAT_VEC}}, addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithMultipleThresholdsGPU) {
  this->RunTestGPU({"RandomBBoxCrop", {"thresholds", "0.0, 0.1, 1.0", DALI_FLOAT_VEC}}, addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithoutThresholdGPU) {
  EXPECT_THROW(this->RunTestGPU({"RandomBBoxCrop", {"thresholds", "", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithThresholdTooLargeFailsGPU) {
  EXPECT_THROW(this->RunTestGPU({"RandomBBoxCrop", {"thresholds", "0.0, 1.1", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithThresholdTooSmallFailsGPU) {
  EXPECT_THROW(this->RunTestGPU({"RandomBBoxCrop", {"thresholds", "-0.1, 1.0", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithThresholdsProvidedInDecreasingOrderGPU) {
  this->RunTestGPU({"RandomBBoxCrop", {"thresholds", "1,0, 0.5, 0.0", DALI_FLOAT_VEC}}, addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithThresholdsRepeatedGPU) {
  this->RunTestGPU({"RandomBBoxCrop", {"thresholds", "1,0, 1.0", DALI_FLOAT_VEC}}, addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithAspectRatioGPU) {
  this->RunTestGPU({"RandomBBoxCrop", {"aspect_ratio", "0.0, 1.0", DALI_FLOAT_VEC}}, addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithSingleAspectRatioGPU) {
  EXPECT_THROW(this->RunTestGPU({"RandomBBoxCrop", {"aspect_ratio", "0.0", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithEmptyAspectRatioGPU) {
  EXPECT_THROW(this->RunTestGPU({"RandomBBoxCrop", {"aspect_ratio", "", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithNegativeAspectRatioGPU) {
  EXPECT_THROW(this->RunTestGPU({"RandomBBoxCrop", {"aspect_ratio", "-0.1, 1.0", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithVeryLargeAspectRatioGPU) {
  this->RunTestGPU({"RandomBBoxCrop", {"aspect_ratio", "0.0, 666.0", DALI_FLOAT_VEC}}, addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithSwappedOrderAspectRatioGPU) {
  EXPECT_THROW(this->RunTestGPU({"RandomBBoxCrop", {"aspect_ratio", "1.0, 0.0", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithScalingGPU) {
  this->RunTestGPU({"RandomBBoxCrop", {"scaling", "0.0, 1.0", DALI_FLOAT_VEC}}, addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithSingleScalingGPU) {
  EXPECT_THROW(this->RunTestGPU({"RandomBBoxCrop", {"scaling", "0.0", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithEmptyScalingGPU) {
  EXPECT_THROW(this->RunTestGPU({"RandomBBoxCrop", {"scaling", "", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithNegativeScalingGPU) {
  EXPECT_THROW(this->RunTestGPU({"RandomBBoxCrop", {"scaling", "-0.1, 1.0", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithVeryLargeScalingGPU) {
  this->RunTestGPU({"RandomBBoxCrop", {"scaling", "0.0, 666.0", DALI_FLOAT_VEC}}, addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithSwappedOrderScalingGPU) {
  EXPECT_THROW(this->RunTestGPU({"RandomBBoxCrop", {"scaling", "1.0, 0.0", DALI_FLOAT_VEC}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithLtrbGPU) {
  this->RunTestGPU({"RandomBBoxCrop", {"ltrb", "true", DALI_BOOL}}, addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithXywhGPU) {
  this->RunTestGPU({"RandomBBoxCrop", {"ltrb", "false", DALI_BOOL}}, addImageType);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithNumAttemptsNegativeGPU) {
  EXPECT_THROW(this->RunTestGPU({"RandomBBoxCrop", {"num_attempts", "-1", DALI_INT32}}, addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithNumAttemptsZeroGPU) {
  EXPECT_THROW(this->RunTestGPU({"RandomBBoxCrop", {"num_attempts", "0", DALI_INT32}},
  addImageType), std::runtime_error);
}

TYPED_TEST(RandomBBoxCropTest, CreateWithNumAttemptsGreaterThanZeroGPU) {
  this->RunTestGPU({"RandomBBoxCrop", {"num_attempts", "1", DALI_INT32}}, addImageType);
}

}  // namespace dali
