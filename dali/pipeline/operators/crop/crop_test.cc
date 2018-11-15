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
TYPED_TEST_CASE(CropTest, Types);

const bool addImageType = true;

TYPED_TEST(CropTest, CropVector) {
  this->RunTest({"Crop", {"crop", "224, 256", DALI_FLOAT_VEC}}, addImageType);
}

}  // namespace dali
