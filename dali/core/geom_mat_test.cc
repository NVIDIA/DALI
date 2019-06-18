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
#include "dali/core/geom/mat.h"

namespace dali {

TEST(Mat, Constructor) {
  mat3x4 m = {{ { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } }};
  for (size_t i = 0; i < m.rows; i++)
    for (size_t j = 0; j < m.cols; j++)
      EXPECT_EQ(m[i][j], i*m.cols+j+1);

  mat<3,1> m31 = vec3(1,2,3);
  EXPECT_EQ(m31(0,0), 1);
  EXPECT_EQ(m31(1,0), 2);
  EXPECT_EQ(m31(2,0), 3);
}

TEST(Mat, Mul) {
  mat3x4 m1 = {{
    { 1, 10, 100, 1000 },
    { 2, 20, 200, 2000 },
    { 4, 40, 400, 4000 },
  }};
  mat<4, 2> m2 = {{
    { 3, 1 },
    { 5, 2 },
    { 7, 3 },
    { 9, 4 }
  }};
  mat3x2 ref = {{
    { 3+50+700+9000,     1+20+300+4000 },
    { 6+100+1400+18000,  2+40+600+8000 },
    { 12+200+2800+36000, 4+80+1200+16000 }
  }};
  mat3x2 result = m1*m2;
  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 2; j++)
      EXPECT_EQ(result(i, j), ref(i, j)) << "@ " << i << ", " << j;
}

TEST(Mat, CompoundAssignScalar) {
  #define TEST_OP(op)\
  {\
    m1 op##= s;\
    for (size_t i = 0; i < 3; i++)\
      for (size_t j = 0; j < 4; j++)\
        EXPECT_EQ(m1(i, j), orig(i, j) op s);\
    m1 = orig;\
  }

  {
    mat3x4 m1 = {{
      { 1, 10, 100, 1000 },
      { 2, 20, 200, 2000 },
      { 4, 40, 400, 4000 },
    }};
    float s = 42;
    auto orig = m1;
    TEST_OP(+)
    TEST_OP(-)
    TEST_OP(*)
    TEST_OP(/)
  }

  {
    imat3x4 m1 = {{
      { 1, 10, 100, 1000 },
      { 2, 20, 200, 2000 },
      { 4, 40, 400, 4000 },
    }};
    int s = 0b1010101;
    auto orig = m1;

    TEST_OP(&)
    TEST_OP(|)
    TEST_OP(^)
  }
}

TEST(Mat, VecTransform) {
  mat2x3 m = {{
    { 2, 3, 100 },
    { 5, 11, 1000 }
  }};
  vec2 v = m * vec3(10, 20, 1);
  EXPECT_EQ(v, vec2(180, 1270));
}

}  // namespace dali
