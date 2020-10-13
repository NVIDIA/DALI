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
#include <random>
#include "dali/test/device_test.h"
#include "dali/core/geom/mat.h"

namespace dali {

TEST(MatTest, Constructor) {
  mat3x4 m = {{ { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } }};
  for (int i = 0; i < m.rows; i++)
    for (int j = 0; j < m.cols; j++)
      EXPECT_EQ(m(i, j), i*m.cols+j+1);

  mat<3, 1> m31 = vec3(1, 2, 3);
  EXPECT_EQ(m31(0, 0), 1);
  EXPECT_EQ(m31(1, 0), 2);
  EXPECT_EQ(m31(2, 0), 3);
}

TEST(MatTest, Equality) {
  mat2x4 a = {{ { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }};
  imat2x4 b = {{ { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }};
  EXPECT_TRUE((a == b));
  EXPECT_FALSE((a != b));
  for (int i = 0; i < a.rows; i++) {
    for (int j = 0; j < a.cols; j++) {
      a(i, j)+=10;
      EXPECT_FALSE((a == b));
      EXPECT_TRUE((a != b));
      b(i, j)+=10;
      EXPECT_TRUE((a == b));
      EXPECT_FALSE((a != b));
    }
  }
}

TEST(MatTest, FromScalar) {
  mat3x4 m = 5;
  mat3x4 ref = {{
    { 5, 0, 0, 0 },
    { 0, 5, 0, 0 },
    { 0, 0, 5, 0 },
  }};
  for (int i = 0; i < m.rows; i++)
    for (int j = 0; j < m.cols; j++)
      EXPECT_EQ(m(i, j), ref(i, j));
}

TEST(MatTest, Transpose) {
  mat3x4 m = {{
    { 1, 2, 3, 4 },
    { 5, 6, 7, 8 },
    { 9, 10, 11, 12 },
  }};
  mat4x3 transposed = m.T();
  for (int i = 0; i < m.rows; i++)
    for (int j = 0; j < m.cols; j++)
      EXPECT_EQ(transposed(j, i), m(i, j));
}

TEST(MatTest, Mul) {
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
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(result(i, j), ref(i, j)) << "@ " << i << ", " << j;
}

TEST(MatTest, Mul4x4) {
  mat4 m1 = {{
    { 1, 10, 100, 1000 },
    { 2, 20, 200, 2000 },
    { 3, 30, 300, 3000 },
    { 4, 40, 400, 4000 },
  }};
  mat4 m2 = {{
    {  1,  2,  3,  4 },
    {  5,  6,  7,  8 },
    {  9, 10, 11, 12 },
    { 13, 14, 15, 16 },
  }};

  mat4 ref = {{
    { 13951, 15062, 16173, 17284 },
    { 27902, 30124, 32346, 34568 },
    { 41853, 45186, 48519, 51852 },
    { 55804, 60248, 64692, 69136 }
  }};
  mat4 result = m1 * m2;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      EXPECT_EQ(result(i, j), ref(i, j)) << "@ " << i << ", " << j;
}


TEST(MatTest, CompoundAssignMatrix) {
  #define TEST_ASSIGN_OP(op)                          \
  {                                                   \
    m1 op##= m2;                                      \
    for (int i = 0; i < 3; i++)                       \
      for (int j = 0; j < 4; j++)                     \
        EXPECT_EQ(m1(i, j), orig(i, j) op m2(i, j));  \
    m1 = orig;                                        \
  }

  {
    mat3x4 m1 = {{
      { 1, 10, 100, 1000 },
      { 2, 20, 200, 2000 },
      { 4, 40, 400, 4000 },
    }};
    mat3x4 m2 = {{
      { 5, 10, 200, 7000 },
      { 6, 90, 300, 6000 },
      { 7, 80, 400, 5000 },
    }};
    auto orig = m1;
    TEST_ASSIGN_OP(+)
    TEST_ASSIGN_OP(-)
  }

  {
    imat3x4 m1 = {{
      { 1, 10, 100, 1324 },
      { 2, 23, 245, 2456 },
      { 4, 45, 413, 4789 },
    }};
    imat3x4 m2 = {{
      { 0, -1, 134, 5423 },
      { 2, 23, 23456, 3, },
      { 4, 45, 413, 4533 },
    }};
    auto orig = m1;

    TEST_ASSIGN_OP(&)
    TEST_ASSIGN_OP(|)
    TEST_ASSIGN_OP(^)
  }
  {
    imat3x4 m1 = {{
      { 1, 10, 100, 1324 },
      { 2, 23, 245, 2456 },
      { 4, 45, 413, 4789 },
    }};
    imat3x4 m2 = {{
      { 0, 1, 2, 3 },
      { 4, 5, 6, 7, },
      { 8, 9, 10, 20 },
    }};
    auto orig = m1;

    TEST_ASSIGN_OP(<<)
    TEST_ASSIGN_OP(>>)
  }
  #undef TEST_ASSIGN_OP
}

TEST(MatTest, CompoundAssignScalar) {
#define TEST_ASSIGN_OP(op)                    \
  {                                           \
    m1 op## = s;                              \
    for (int i = 0; i < 3; i++)               \
      for (int j = 0; j < 4; j++)             \
        EXPECT_EQ(m1(i, j), orig(i, j) op s); \
    m1 = orig;                                \
  }

  {
    mat3x4 m1 = {{
      { 1, 10, 100, 1000 },
      { 2, 20, 200, 2000 },
      { 4, 40, 400, 4000 },
    }};
    float s = 42;
    auto orig = m1;
    TEST_ASSIGN_OP(+)
    TEST_ASSIGN_OP(-)
    TEST_ASSIGN_OP(*)
    TEST_ASSIGN_OP(/)
  }

  {
    imat3x4 m1 = {{
      { 1, 10, 100, 1000 },
      { 2, 20, 200, 2000 },
      { 4, 40, 400, 4000 },
    }};
    int s = 0b1010101;
    auto orig = m1;

    TEST_ASSIGN_OP(&)
    TEST_ASSIGN_OP(|)
    TEST_ASSIGN_OP(^)

    s = 5;
    TEST_ASSIGN_OP(<<)
    TEST_ASSIGN_OP(>>)
  }
  #undef TEST_ASSIGN_OP
}

struct cuda_rng {
  DALI_HOST_DEV int operator()() {
    return seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  }
  unsigned seed;
};

template <int rows, int cols, typename T, typename RNG>
DALI_HOST_DEV
void RandomFill(mat<rows, cols, T> &m, RNG &rng, int lo, int hi) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      m(i, j) = rng()%(hi-lo) + lo;
}

TEST(MatTest, BinOp) {
#define TEST_OPERATOR_SIZE(op, mat_type) {                    \
    mat_type m1, m2;                                          \
    int lo = 0, hi = 31;                                      \
    cuda_rng rng = { 12345 };                                 \
    RandomFill(m1, rng, lo, hi); RandomFill(m2, rng, lo, hi); \
    mat_type result = m1 op m2;                               \
    for (int i = 0; i < result.rows; i++)                     \
      for (int j = 0; j < result.cols; j++)                   \
        EXPECT_EQ(result(i, j), m1(i, j) op m2(i, j));        \
    int scalar = rng()&31;                                    \
    result = m1 op scalar;                                    \
    for (int i = 0; i < result.rows; i++)                     \
      for (int j = 0; j < result.cols; j++)                   \
        EXPECT_EQ(result(i, j), m1(i, j) op scalar);          \
    result = scalar op m2;                                    \
    for (int i = 0; i < result.rows; i++)                     \
      for (int j = 0; j < result.cols; j++)                   \
        EXPECT_EQ(result(i, j), scalar op m2(i, j));          \
  }

#define TEST_OPERATOR_TYPE(op, prefix)   \
  TEST_OPERATOR_SIZE(op, prefix##mat2)   \
  TEST_OPERATOR_SIZE(op, prefix##mat3)   \
  TEST_OPERATOR_SIZE(op, prefix##mat4)   \
  TEST_OPERATOR_SIZE(op, prefix##mat2x3) \
  TEST_OPERATOR_SIZE(op, prefix##mat3x2) \
  TEST_OPERATOR_SIZE(op, prefix##mat3x4) \
  TEST_OPERATOR_SIZE(op, prefix##mat4x3) \
  TEST_OPERATOR_SIZE(op, prefix##mat2x4) \
  TEST_OPERATOR_SIZE(op, prefix##mat4x2)

#define TEST_OPERATOR(op)             \
  TEST_OPERATOR_TYPE(op, /* empty */) \
  TEST_OPERATOR_TYPE(op, i)

  TEST_OPERATOR(+)
  TEST_OPERATOR(-)
  TEST_OPERATOR_TYPE(|, i)
  TEST_OPERATOR_TYPE(&, i)
  TEST_OPERATOR_TYPE(^, i)
  TEST_OPERATOR_TYPE(<<, i)
  TEST_OPERATOR_TYPE(>>, i)
}

DEVICE_TEST(Dev_Mat, NormalUsage, 1, 1) {
  mat4 m = {{
    { 2, 1, 0, 0 },
    { 1, 2, 0, 0 },
    { 0, 0, 1, 0 },
    { 0, 0, 0, 1 }
  }};

  vec4 v = { 1, 2, 3, 1 };
  v = m*v;
  DEV_EXPECT_EQ(v[0], 4);
  DEV_EXPECT_EQ(v[1], 5);
  DEV_EXPECT_EQ(v[2], 3);
  DEV_EXPECT_EQ(v[3], 1);
}

TEST(MatTest, VecTransform) {
  mat2x3 m = {{
    { 2, 3, 100 },
    { 5, 11, 1000 }
  }};
  vec2 v = m * vec3(10, 20, 1);
  EXPECT_EQ(v, vec2(180, 1270));
}

TEST(MatTest, SelectCol) {
  mat3 m = {{
    { 1, 2, 3 },
    { 4, 5, 6 },
    { 7, 8, 9 }
  }};
  mat3x1 col = sub<3, 1>(m, 0, 1);
  vec3 v = col;
  EXPECT_EQ(v, vec3(2, 5, 8));
}

TEST(MatTest, CatCols) {
  mat3 a = {{
    { 1, 2, 3 },
    { 4, 5, 6 },
    { 7, 8, 9 }
  }};
  mat3x2 b = {{
    { 10, 20 },
    { 30, 40 },
    { 50, 60 }
  }};
  mat<3, 5> result = cat_cols(a, b);
  mat<3, 5> ref = {{
    { 1, 2, 3, 10, 20 },
    { 4, 5, 6, 30, 40 },
    { 7, 8, 9, 50, 60 }
  }};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 5; j++)
      EXPECT_EQ(result(i, j), ref(i, j)) << "@ " << i << ", " << j;
}

TEST(MatTest, CatColsVec) {
  mat3 a = {{
    { 1, 2, 3 },
    { 4, 5, 6 },
    { 7, 8, 9 }
  }};
  vec3 b = { 10, 20, 30 };
  vec3 c = { 40, 50, 60 };
  mat3x4 result = cat_cols(a, b);
  mat3x4 ref = {{
    { 1, 2, 3, 10 },
    { 4, 5, 6, 20 },
    { 7, 8, 9, 30 }
  }};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      EXPECT_EQ(result(i, j), ref(i, j)) << "@ " << i << ", " << j;

  result = cat_cols(b, a);
  ref = mat3x4{{
    { 10, 1, 2, 3 },
    { 20, 4, 5, 6 },
    { 30, 7, 8, 9 }
  }};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      EXPECT_EQ(result(i, j), ref(i, j)) << "@ " << i << ", " << j;

  mat3x2 result2 = cat_cols(b, c);
  mat3x2 ref2 = {{
    { 10, 40 },
    { 20, 50 },
    { 30, 60 }
  }};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_EQ(result2(i, j), ref2(i, j)) << "@ " << i << ", " << j;
}

TEST(MatTest, CatRows) {
  mat3 a = {{
    { 1, 2, 3 },
    { 4, 5, 6 },
    { 7, 8, 9 }
  }};
  mat2x3 b = {{
    { 10, 20, 30 },
    { 40, 50, 60 }
  }};
  mat<5, 3> result = cat_rows(a, b);
  mat<5, 3> ref = {{
    { 1, 2, 3 },
    { 4, 5, 6 },
    { 7, 8, 9 },
    { 10, 20, 30 },
    { 40, 50, 60 }
  }};
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 3; j++)
      EXPECT_EQ(result(i, j), ref(i, j)) << "@ " << i << ", " << j;
}


TEST(MatTest, IdentityMatrix) {
  mat3 three_by_three = {{
         {1, 0, 0},
         {0, 1, 0},
         {0, 0, 1}
  }};
  mat3x1 three_by_one = {{
         {1}, {0}, {0}
  }};
  mat3x2 three_by_two = {{
         {1, 0},
         {0, 1},
         {0, 0},
  }};
  auto m1 = imat3::eye();
  auto m2 = mat<3, 1, int>::eye();
  auto m3 = mat<3, 2, int>::eye();
  EXPECT_EQ(m1, three_by_three);
  EXPECT_EQ(m2, three_by_one);
  EXPECT_EQ(m3, three_by_two);
}

TEST(MatTest, SolveGauss) {
  mat4 system_l = {{
    {0, 1, 2, 1},
    {1, 0, 1, 0},
    {1, 1, 0, 1},
    {1, 1, 1, 0}
  }};
  mat4x1 system_r = {{
    {3}, {3}, {3}, {3}
  }};
  const mat4x1 solution = {{
    {2}, {0}, {1}, {1}
  }};
  solve_gauss(system_l, system_r);
  EXPECT_EQ(system_l, mat4::eye());
  EXPECT_EQ(system_r, solution);
}

TEST(MatTest, Inverse) {
  #define EXPECT_MAT_EQ(m1, m2) \
  for (int r = 0; r < m1.rows; ++r)  \
    for (int c = 0; c < m1.cols; ++c)  \
      EXPECT_FLOAT_EQ(m1(r, c), m2(r, c));
  mat2 two_by_two = {{
    {3, 5},
    {4, 7}
  }};
  mat2 inv2x2 = {{
    {7, -5},
    {-4, 3}
  }};
  EXPECT_EQ(inverse(two_by_two), inv2x2);

  mat3 three_by_three = {{
    {0.5, 0.5, 0  },
    {0.5, 0.5, 1  },
    {0.5, 0,   0.5}
  }};
  mat3 inv3x3 = {{
    { 1, -1,  2},
    { 1,  1, -2},
    {-1,  1,  0}
  }};
  EXPECT_EQ(inverse(three_by_three), inv3x3);

  mat3 three_by_three2 = {{
    {1, 5, 7},
    {1, 3, 6},
    {2, 4, 1}
  }};

  mat3 inv3x3_2 = {{
    {-1.05,  1.15, 0.45},
    { 0.55, -0.65, 0.05},
    {-0.1,   0.3, -0.1 }
  }};
  EXPECT_MAT_EQ(inverse(three_by_three2), inv3x3_2);
}

DEVICE_TEST(MatTest, DevInverse, 1, 1) {
  #define DEV_EXPECT_MAT_EQ(m1, m2) \
  for (int r = 0; r < m1.rows; ++r)  \
    for (int c = 0; c < m1.cols; ++c)  \
      DEV_EXPECT_LT(abs(m1(r, c) - m2(r, c)), 1e-6f);

  mat2 two_by_two = {{
    {3, 5},
    {4, 7}
  }};
  mat2 inv2x2 = {{
    {7, -5},
    {-4, 3}
  }};
  DEV_EXPECT_MAT_EQ(inverse(two_by_two), inv2x2);

  mat3 three_by_three = {{
    {0.5, 0.5, 0  },
    {0.5, 0.5, 1  },
    {0.5, 0,   0.5}
  }};
  mat3 inv3x3 = {{
    { 1, -1,  2},
    { 1,  1, -2},
    {-1,  1,  0}
  }};
  DEV_EXPECT_MAT_EQ(inverse(three_by_three), inv3x3);

  mat3 three_by_three2 = {{
    {1, 5, 7},
    {1, 3, 6},
    {2, 4, 1}
  }};

  mat3 inv3x3_2 = {{
    {-1.05,  1.15, 0.45},
    { 0.55, -0.65, 0.05},
    {-0.1,   0.3, -0.1 }
  }};
  DEV_EXPECT_MAT_EQ(inverse(three_by_three2), inv3x3_2);
}

TEST(MatTest, InverseFail) {
  mat2 m = {{
    {1, 2},
    {1, 2}
  }};
  EXPECT_THROW(inverse(m), std::range_error);
}

TEST(MatTest, Print) {
  mat4x3 m = {{
    { 1, 2, 3 },
    { 1.5, 2.5, 3.5 },
    { -2, -3, -4 },
    { 0, 1e+30f, 1 }
  }};
  std::stringstream ss;
  ss << m;
  EXPECT_EQ(ss.str(),
"|   1      2    3 |\n"
"| 1.5    2.5  3.5 |\n"
"|  -2     -3   -4 |\n"
"|   0  1e+30    1 |");
}

}  // namespace dali
