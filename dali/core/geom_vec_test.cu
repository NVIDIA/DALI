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
#include "dali/test/device_test.h"
#include "dali/core/geom/vec.h"

namespace dali {
template <size_t N, typename T>
__device__ DeviceString dev_to_string(const vec<N, T> &v) {
  DeviceString str;
  for (size_t i = 0; i < N; i++) {
    if (i) str += ", ";
    str += dev_to_string(v[i]);
  }
  return str;
}

static_assert(sizeof(vec<1, float>)   == 1*sizeof(float),   "Invalid size for a vector");
static_assert(sizeof(vec<2, int16_t>) == 2*sizeof(int16_t), "Invalid size for a vector");
static_assert(sizeof(vec<2, float>)   == 2*sizeof(float),   "Invalid size for a vector");
static_assert(sizeof(vec<3, char>)    == 3*sizeof(char),    "Invalid size for a vector");
static_assert(sizeof(vec<4, float>)   == 4*sizeof(float),   "Invalid size for a vector");
static_assert(sizeof(vec<5, float>)   == 5*sizeof(float),   "Invalid size for a vector");

TEST(Vec, BraceConstruct) {
  vec<1> v1 = {};
  EXPECT_EQ(v1.x, 0) << "vec should be zero-initialized by default";
  vec<2> v2 = {};
  EXPECT_EQ(v2.x, 0) << "vec should be zero-initialized by default";
  EXPECT_EQ(v2.x, 0) << "vec should be zero-initialized by default";
  vec<3> v3 = {};
  EXPECT_EQ(v3.x, 0) << "vec should be zero-initialized by default";
  EXPECT_EQ(v3.y, 0) << "vec should be zero-initialized by default";
  EXPECT_EQ(v3.z, 0) << "vec should be zero-initialized by default";
  vec<4> v4 = {};
  EXPECT_EQ(v4[0], 0) << "vec should be zero-initialized by default";
  EXPECT_EQ(v4[1], 0) << "vec should be zero-initialized by default";
  EXPECT_EQ(v4[2], 0) << "vec should be zero-initialized by default";
  EXPECT_EQ(v4[3], 0) << "vec should be zero-initialized by default";
  vec<5> v5 = {};
  EXPECT_EQ(v5[0], 0) << "vec should be zero-initialized by default";
  EXPECT_EQ(v5[1], 0) << "vec should be zero-initialized by default";
  EXPECT_EQ(v5[2], 0) << "vec should be zero-initialized by default";
  EXPECT_EQ(v5[3], 0) << "vec should be zero-initialized by default";
  EXPECT_EQ(v5[4], 0) << "vec should be zero-initialized by default";
}

TEST(Vec, FieldConstruct) {
  vec<1> v1 = { 1 };
  EXPECT_EQ(v1.x, 1);
  EXPECT_EQ(v1[0], 1);
  vec<2> v2 = { 1, 2 };
  EXPECT_EQ(v2.x, 1);
  EXPECT_EQ(v2.y, 2);
  EXPECT_EQ(v2[0], 1);
  EXPECT_EQ(v2[1], 2);
  vec<3> v3 = { 1, 2, 3 };
  EXPECT_EQ(v3.x, 1);
  EXPECT_EQ(v3.y, 2);
  EXPECT_EQ(v3.z, 3);
  EXPECT_EQ(v3[2], 3);
  vec<4> v4 = { 1, 2, 3, 4 };
  EXPECT_EQ(v4.x, 1);
  EXPECT_EQ(v4.y, 2);
  EXPECT_EQ(v4.z, 3);
  EXPECT_EQ(v4.w, 4);
  EXPECT_EQ(v4[0], 1);
  EXPECT_EQ(v4[1], 2);
  EXPECT_EQ(v4[2], 3);
  EXPECT_EQ(v4[3], 4);
  vec<5> v5 = { 1, 2, 3, 4, 5 };
  EXPECT_EQ(v5[0], 1);
  EXPECT_EQ(v5[1], 2);
  EXPECT_EQ(v5[2], 3);
  EXPECT_EQ(v5[3], 4);
  EXPECT_EQ(v5[4], 5);
}

TEST(Vec, Equality) {
  vec<3> a = { 1, 2, 3 };
  vec<3, int> b = { 1, 2, 3 };
  vec<3> c = { 1, 2, 4 };
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a == c);
  EXPECT_FALSE(a != b);
  EXPECT_TRUE(a != c);
}

TEST(Vec, Op) {
  vec<3> a = { 1, 2, 3 }, b = { 4, 5, 6 };
  EXPECT_EQ(a+b, (vec<3>{5, 7, 9}));
  EXPECT_EQ(a-b, (vec<3>{-3, -3, -3}));
  EXPECT_EQ(a*b, (vec<3>{4, 10, 18}));
  b = { 3, 2, 1 };
  EXPECT_EQ((a<b), (vec<3, bool>(true, false, false)));
  EXPECT_EQ((b<a), (vec<3, bool>(false, false, true)));
}

DEVICE_TEST(Dev_Vec, Op, 1, 1) {
  vec<3> a = { 1, 2, 3 }, b = { 4, 5, 6 };
  DEV_EXPECT_EQ(a+b, (vec<3>{5, 7, 9}));
  DEV_EXPECT_EQ(a-b, (vec<3>{-3, -3, -3}));
  DEV_EXPECT_EQ(a*b, (vec<3>{4, 10, 18}));
  b = { 3, 2, 1 };
  DEV_EXPECT_EQ((a<b), (vec<3, bool>(true, false, false)));
  DEV_EXPECT_EQ((b<a), (vec<3, bool>(false, false, true)));
}

TEST(Vec, Cast) {
  vec<4> a = { 1.2f, 2.4f, 3.4f, -5.3f };
  EXPECT_EQ(a.cast<int>(), cast<int>(a));
  EXPECT_EQ(a.cast<int>(), (vec<4, int>(1, 2, 3, -5)));
}

TEST(Vec, Iteration) {
  const size_t N = 3;
  vec<N> v;
  EXPECT_EQ(dali::size(v), N);
  for (size_t i = 0; i < N; i++)
    v[i] = i + 5;
  for (auto &x : v) {
    auto ofs = &x - &v[0];
    ASSERT_TRUE(ofs >= 0 && ofs < static_cast<ptrdiff_t>(N));
    x++;
  }

  for (size_t i = 0; i < N; i++)
    EXPECT_EQ(v[i], i + 6);
}

TEST(Vec, Dot) {
  vec<3> a = { 1, 10, 100 }, b = { 2, 3, 4 };
  EXPECT_EQ(dot(a, b), 432);
}

TEST(Vec, RoundInt) {
  vec<3> f = { -0.6f, 0.1f, 0.7f };
  auto i = round_int(f);
  EXPECT_EQ(i, (vec<3, int>(-1, 0, 1)));
}

DEVICE_TEST(Dev_Vec, RoundInt, 1, 1) {
  vec<3> f = { -0.6f, 0.1f, 0.7f };
  auto i = round_int(f);
  DEV_EXPECT_EQ(i, (vec<3, int>(-1, 0, 1)));
}

DEVICE_TEST(Dev_Vec, Cat, 1, 1) {
  vec<3> a = { 1, 2, 3 };
  vec<2> b = { 4, 5 };
  DEV_EXPECT_EQ(cat(a, b), (vec<5>(1, 2, 3, 4, 5)));
  DEV_EXPECT_EQ(cat(b, a), (vec<5>(4, 5, 1, 2, 3)));
  DEV_EXPECT_EQ(cat(a, 4.0f), vec4(1, 2, 3, 4));
  DEV_EXPECT_EQ(cat(0.5f, b), vec3(0.5f, 4, 5));
}

DEVICE_TEST(Dev_Vec, OpScalar, 1, 1) {
  vec4 v = { -1, 1, 2, 3 };
  vec4 v1 = v * 2;
  vec4 v2 = v / 2;
  vec4 v3 = v + 0.5f;
  vec4 v4 = v - 0.5f;
  vec4 v5 = 0.5f - v;
  DEV_EXPECT_EQ(v1, vec4(-2, 2, 4, 6));
  DEV_EXPECT_EQ(v2, vec4(-0.5f, 0.5f, 1.0f, 1.5f));
  DEV_EXPECT_EQ(v3, vec4(-0.5f, 1.5f, 2.5f, 3.5f));
  DEV_EXPECT_EQ(v4, vec4(-1.5f, 0.5f, 1.5f, 2.5f));
  DEV_EXPECT_EQ(v5, vec4(1.5f, -0.5f, -1.5f, -2.5f));
}

DEVICE_TEST(Dev_Vec, OpAssignScalar, 1, 1) {
  vec4 v;
  v = 3;
  DEV_EXPECT_EQ(v, vec4(3,3,3,3));
  v = { 1, 2, 3, 4 };
  v += 1;
  DEV_EXPECT_EQ(v, vec4(2,3,4,5));
}

TEST(Vec, Promote) {
  vec4 vi = { 1, 2, 3, 4 };
  vec4 vf = vi + 0.5f;
  EXPECT_EQ(vf, vec4(1.5f, 2.5f, 3.5f, 4.5f));
  vf = vi + vf;
  EXPECT_EQ(vf, vec4(2.5f, 4.5f, 6.5f, 8.5f));
  auto minus = 0-i8vec4(1, 2, 3, 4);
  static_assert(std::is_same<decltype(minus)::element_t, int>::value,
    "Wrong element type inferred. Should be int.");
  EXPECT_EQ(minus, ivec4(-1,-2,-3,-4));
}

}  // namespace dali
