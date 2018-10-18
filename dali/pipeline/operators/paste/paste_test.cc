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
class PasteTest : public GenericMatchingTest<ImgType> {
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(PasteTest, Types);

static const char *opName = "Paste";
const bool addImageType = true;

TYPED_TEST(PasteTest, Test1) {
  const OpArg params[] = {{"ratio", "2.", DALI_FLOAT},
                          {"fill_value", "55, 155, 155", DALI_INT_VEC},
                          {"paste_y", "0.1", DALI_FLOAT}};

  this->RunTest(opName, params, sizeof(params)/sizeof(params[0]), addImageType);
}


TYPED_TEST(PasteTest, Test2) {
  const OpArg params[] = {{"ratio", "1.5", DALI_FLOAT},
                          {"fill_value", "155, 10, 15", DALI_INT_VEC},
                          {"paste_y", "0.2", DALI_FLOAT},
                          {"paste_x", ".4", DALI_FLOAT }};

  this->RunTest(opName, params, sizeof(params)/sizeof(params[0]), addImageType);
}

// paste_x = .5, paste_y = .4)
}  // namespace dali
