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
#include "dali/core/math_util.h"

namespace dali {

TEST(MathUtil, rsqrt) {
  EXPECT_NEAR(rsqrt(0.25f), 2.0f, 1e-6f);
  EXPECT_NEAR(rsqrt(0.16f), 2.5f, 1e-6f);
  EXPECT_NEAR(rsqrt(16.0f), 0.25f, 1e-7f);
}

TEST(MathUtil, rsqrtd) {
  EXPECT_NEAR(rsqrt(0.25), 2.0, 1e-11);
  EXPECT_NEAR(rsqrt(0.16), 2.5, 1e-11);
  EXPECT_NEAR(rsqrt(16.0), 0.25, 1e-12);
}

}  // namespace dali
