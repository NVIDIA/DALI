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
class ColorTest : public GenericMatchingTest<ImgType> {
};

typedef ::testing::Types<RGB> Types;
TYPED_TEST_SUITE(ColorTest, Types);

TYPED_TEST(ColorTest, Brightness) {
  this->RunTest({"Brightness", {"brightness", "3.", DALI_FLOAT}, 1e-4});
}

TYPED_TEST(ColorTest, Contrast) {
  this->RunTest({"Contrast", {"contrast", "1.3", DALI_FLOAT}, 0.18});
}

TYPED_TEST(ColorTest, Saturation) {
  this->RunTest({"Saturation", {"saturation", "3.", DALI_FLOAT}, 0.33});
}

TYPED_TEST(ColorTest, Hue) {
  this->RunTest({"Hue", {"hue", "31.456", DALI_FLOAT}, 0.39});
}

}  // namespace dali
