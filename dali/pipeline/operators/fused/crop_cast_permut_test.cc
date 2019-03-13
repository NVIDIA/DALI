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
#include "dali/test/dali_test_utils.h"

namespace dali {

template <typename ImgType>
class CropCastPermuteTest : public GenericMatchingTest<ImgType> {};


typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_SUITE(CropCastPermuteTest, Types);

const bool addImageType = true;

TYPED_TEST(CropCastPermuteTest, CropVector) {
  this->RunTest({"CropCastPermute", {"crop", "224, 256", DALI_FLOAT_VEC}}, addImageType);
}

TYPED_TEST(CropCastPermuteTest, Layout_DALI_NCHW) {
  const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                          {"output_layout", EnumToString(DALI_NCHW), DALI_INT32}};
  this->RunTest("CropCastPermute", params, sizeof(params) / sizeof(params[0]), addImageType);
}

TYPED_TEST(CropCastPermuteTest, Layout_DALI_NHWC) {
  const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                          {"output_layout", EnumToString(DALI_NHWC), DALI_INT32}};
  this->RunTest("CropCastPermute", params, sizeof(params) / sizeof(params[0]), addImageType);
}

TYPED_TEST(CropCastPermuteTest, Layout_DALI_SAME) {
  const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                          {"output_layout", EnumToString(DALI_SAME), DALI_INT32}};
  this->RunTest("CropCastPermute", params, sizeof(params) / sizeof(params[0]), addImageType);
}

TYPED_TEST(CropCastPermuteTest, Output_DALI_NO_TYPE) {
  const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                          {"output_dtype", EnumToString(DALI_NO_TYPE), DALI_INT32}};
  this->RunTest("CropCastPermute", params, sizeof(params) / sizeof(params[0]), addImageType);
}

TYPED_TEST(CropCastPermuteTest, Output_DALI_UINT8) {
  const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                          {"output_dtype", EnumToString(DALI_UINT8), DALI_INT32}};
  this->RunTest("CropCastPermute", params, sizeof(params) / sizeof(params[0]), addImageType);
}

TYPED_TEST(CropCastPermuteTest, Output_DALI_INT16) {
  const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                          {"output_dtype", EnumToString(DALI_INT16), DALI_INT32}};
  this->RunTest("CropCastPermute", params, sizeof(params) / sizeof(params[0]), addImageType);
}

TYPED_TEST(CropCastPermuteTest, Output_DALI_INT32) {
  const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                          {"output_dtype", EnumToString(DALI_INT32), DALI_INT32}};
  this->RunTest("CropCastPermute", params, sizeof(params) / sizeof(params[0]), addImageType);
}

TYPED_TEST(CropCastPermuteTest, Output_DALI_INT64) {
  const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                          {"output_dtype", EnumToString(DALI_INT64), DALI_INT32}};
  this->RunTest("CropCastPermute", params, sizeof(params) / sizeof(params[0]), addImageType);
}

TYPED_TEST(CropCastPermuteTest, Output_DALI_FLOAT16) {
  const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                          {"output_dtype", EnumToString(DALI_FLOAT16), DALI_INT32}};
  this->RunTest("CropCastPermute", params, sizeof(params) / sizeof(params[0]), addImageType);
}

TYPED_TEST(CropCastPermuteTest, Output_DALI_FLOAT) {
  const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                          {"output_dtype", EnumToString(DALI_FLOAT), DALI_INT32}};
  this->RunTest("CropCastPermute", params, sizeof(params) / sizeof(params[0]), addImageType);
}

}  // namespace dali
