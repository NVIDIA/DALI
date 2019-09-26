// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <gtest/gtest.h>
#include "dali/pipeline/operators/color/hsv.h"

namespace dali {

namespace testing {

TEST(HsvTest, transformation_matrix) {
    auto identity = hsv::transformation_matrix(0, 1, 1);
    auto one = eye<3, 3>();
    for (int i = 0; i < identity.cols; i++) {
        for (int j = 0; j < identity.rows; j++) {
            EXPECT_NEAR(identity(i, j), one(i, j), .01f);
        }
    }
}

}  // namespace testing
}  // namespace dali
