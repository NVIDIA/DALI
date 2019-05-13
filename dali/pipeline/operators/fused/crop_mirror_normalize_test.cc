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

static const char *opName = "CropMirrorNormalize";

template <typename ImgType>
class CropMirrorNormalizeTest : public GenericMatchingTest<ImgType> {
};

const float eps = 50000;
const bool addImageType = true;

static const OpArg vector_params[] = {{"crop", "224, 256", DALI_FLOAT_VEC},
                                      {"mean", "0.", DALI_FLOAT_VEC},
                                      {"std", "1.", DALI_FLOAT_VEC}};

typedef ::testing::Types<RGB, BGR, Gray> Types;

TYPED_TEST_SUITE(CropMirrorNormalizeTest, Types);

TYPED_TEST(CropMirrorNormalizeTest, CropVector) {
  this->RunTest(opName, vector_params,
                sizeof(vector_params) / sizeof(vector_params[0]), addImageType,
                eps);
}

TYPED_TEST(CropMirrorNormalizeTest, Layout_DALI_NCHW) {
  static const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                                 {"mean", "0.", DALI_FLOAT_VEC},
                                 {"std", "1.", DALI_FLOAT_VEC},
                                 {"output_layout", EnumToString(DALI_NCHW), DALI_INT32}};

  this->RunTest(opName, params, sizeof(params) / sizeof(params[0]),
                addImageType, eps);
}

TYPED_TEST(CropMirrorNormalizeTest, Layout_DALI_NHWC) {
  static const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                                 {"mean", "0.", DALI_FLOAT_VEC},
                                 {"std", "1.", DALI_FLOAT_VEC},
                                 {"output_layout", EnumToString(DALI_NHWC), DALI_INT32}};

  this->RunTest(opName, params, sizeof(params) / sizeof(params[0]),
                addImageType, eps);
}

TYPED_TEST(CropMirrorNormalizeTest, Layout_DALI_SAME) {
  static const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                                 {"mean", "0.", DALI_FLOAT_VEC},
                                 {"std", "1.", DALI_FLOAT_VEC},
                                 {"output_layout", EnumToString(DALI_SAME), DALI_INT32}};

  this->RunTest(opName, params, sizeof(params) / sizeof(params[0]),
                addImageType, eps);
}

TYPED_TEST(CropMirrorNormalizeTest, Output_DALI_NO_TYPE) {
  static const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                                 {"mean", "0.", DALI_FLOAT_VEC},
                                 {"std", "1.", DALI_FLOAT_VEC},
                                 {"output_dtype", "-1", DALI_INT32}};

  this->RunTest(opName, params, sizeof(params) / sizeof(params[0]),
                addImageType, eps);
}

TYPED_TEST(CropMirrorNormalizeTest, Output_DALI_UINT8) {
  static const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                                 {"mean", "0.", DALI_FLOAT_VEC},
                                 {"std", "1.", DALI_FLOAT_VEC},
                                 {"output_dtype", "0", DALI_INT32}};

  this->RunTest(opName, params, sizeof(params) / sizeof(params[0]),
                addImageType, eps);
}

TYPED_TEST(CropMirrorNormalizeTest, Output_DALI_INT16) {
  static const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                                 {"mean", "0.", DALI_FLOAT_VEC},
                                 {"std", "1.", DALI_FLOAT_VEC},
                                 {"output_dtype", "1", DALI_INT32}};

  this->RunTest(opName, params, sizeof(params) / sizeof(params[0]),
                addImageType, eps);
}

TYPED_TEST(CropMirrorNormalizeTest, Output_DALI_INT32) {
  static const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                                 {"mean", "0.", DALI_FLOAT_VEC},
                                 {"std", "1.", DALI_FLOAT_VEC},
                                 {"output_dtype", "2", DALI_INT32}};

  this->RunTest(opName, params, sizeof(params) / sizeof(params[0]),
                addImageType, eps);
}

TYPED_TEST(CropMirrorNormalizeTest, Output_DALI_INT64) {
  static const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                                 {"mean", "0.", DALI_FLOAT_VEC},
                                 {"std", "1.", DALI_FLOAT_VEC},
                                 {"output_dtype", "3", DALI_INT32}};

  this->RunTest(opName, params, sizeof(params) / sizeof(params[0]),
                addImageType, eps);
}

TYPED_TEST(CropMirrorNormalizeTest, Output_DALI_FLOAT16) {
  static const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                                 {"mean", "0.", DALI_FLOAT_VEC},
                                 {"std", "1.", DALI_FLOAT_VEC},
                                 {"output_dtype", "4", DALI_INT32}};

  this->RunTest(opName, params, sizeof(params) / sizeof(params[0]),
                addImageType, eps);
}

TYPED_TEST(CropMirrorNormalizeTest, Output_DALI_FLOAT) {
  static const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                                 {"mean", "0.", DALI_FLOAT_VEC},
                                 {"std", "1.", DALI_FLOAT_VEC},
                                 {"output_dtype", "5", DALI_INT32}};

  this->RunTest(opName, params, sizeof(params) / sizeof(params[0]),
                addImageType, eps);
}

}  // namespace dali
