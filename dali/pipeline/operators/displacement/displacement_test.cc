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
class DisplacementTest : public GenericMatchingTest<ImgType> {
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_SUITE(DisplacementTest, Types);

TYPED_TEST(DisplacementTest, Sphere) {
  this->RunTest("Sphere");
}

TYPED_TEST(DisplacementTest, Water) {
  const OpArg params[] = {{"ampl_x", "2.", DALI_FLOAT},
                          {"ampl_y", "3.", DALI_FLOAT},
                          {"phase_x", "0.2", DALI_FLOAT}};
  this->RunTest("Water", params, sizeof(params)/sizeof(params[0]));
}

/*
 * As of 08/03/2018 this test is disabled because Jitter is not activated for CPU
 *
TYPED_TEST(DisplacementTest, Jitter) {
  this->RunTest("Jitter");
}
*/

TYPED_TEST(DisplacementTest, WarpAffine) {
  const OpArg params = {"matrix", "1.0, 0.8, 0.0, 0.0, 1.2, 0.0", DALI_FLOAT_VEC};
  this->RunTest("WarpAffine", &params, 1);
}

TYPED_TEST(DisplacementTest, Rotate) {
  this->RunTest({"Rotate", {"angle", "10", DALI_FLOAT}, 0.001});
}

TYPED_TEST(DisplacementTest, Flip) {
  const OpArg params[] = {{"horizontal", "1", DALI_INT32},
                          {"vertical", "1", DALI_INT32}};
  this->RunTest("Flip", params, 2);
}

}  // namespace dali
