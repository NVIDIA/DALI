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
#include "dali/core/geom/transform.h"

namespace dali {

TEST(GeomTransform, Translation2D) {
  vec2 offset = { 1.5f, -0.5f };
  vec3 translated = translation(offset) * vec3(5, 6, 1);
  EXPECT_EQ(translated, vec3(6.5f, 5.5f, 1.0f));
}

TEST(GeomTransform, Translation3D) {
  vec3 offset = { -1.5f, 0.5f, 0.25f };
  vec4 translated = translation(offset) * vec4(5, 6, 7, 1);
  EXPECT_EQ(translated, vec4(3.5f, 6.5f, 7.25f, 1.0f));
}

TEST(GeomTransform, Scaling2D) {
  vec3 scaled = scaling({ 2, 3 }) * vec3(3, 5, 1);
  EXPECT_EQ(scaled, vec3(6, 15, 1));
}

TEST(GeomTransform, Scaling3D) {
  vec4 scaled = scaling({ 2, 3, 4 }) * vec4(3, 5, 7, 1);
  EXPECT_EQ(scaled, vec4(6, 15, 28, 1));
}

TEST(GeomTransform, Rotation2D) {
  vec3 rotated1 = rotation2D(M_PI/4) * vec3(1, 0, 1);
  vec3 rotated2 = rotation2D(M_PI/4) * vec3(0, 1, 1);
  vec3 rotated3 = rotation2D(M_PI/4) * vec3(1, 1, 1);
  vec3 rotated4 = rotation2D(M_PI/4) * vec3(1, -1, 1);
  EXPECT_EQ(rotated1[2], 1) << "Homogeneous vector expected";
  EXPECT_EQ(rotated2[2], 1) << "Homogeneous vector expected";
  EXPECT_EQ(rotated3[2], 1) << "Homogeneous vector expected";
  EXPECT_EQ(rotated4[2], 1) << "Homogeneous vector expected";

  constexpr float s2    = M_SQRT2;
  constexpr float shalf = M_SQRT1_2;

  EXPECT_NEAR(rotated1.x, shalf, 1e-6f);
  EXPECT_NEAR(rotated1.y, shalf, 1e-6f);

  EXPECT_NEAR(rotated2.x, -shalf, 1e-6f);
  EXPECT_NEAR(rotated2.y, shalf,  1e-6f);

  EXPECT_NEAR(rotated3.x, 0,  1e-6f);
  EXPECT_NEAR(rotated3.y, s2, 1e-6f);

  EXPECT_NEAR(rotated4.x, s2, 1e-6f);
  EXPECT_NEAR(rotated4.y, 0,  1e-6f);
}

struct vec_near {
  template <int N, typename T>
  inline bool operator()(vec<N, T> a, vec<N, T> b, double eps) const {
    for (int i = 0; i < N; i++)
      if (std::abs(a[i] - b[i]) > eps)
        return false;
    return true;
  }
};

#define EXPECT_VEC_NEAR(v1, v2, eps) EXPECT_PRED3(vec_near(), v1, v2, eps)

TEST(GeomTransform, Rotation3D_MainAxes) {
  const float a = M_PI/2;
  vec4 r4;
  vec3 rotated;
  r4 = rotation3D({0, 0, 1}, a) * vec4(1, 2, 3, 1);
  EXPECT_EQ(r4.w, 1.0f);
  rotated = sub<3>(r4);
  EXPECT_VEC_NEAR(rotated, vec3(-2, 1, 3), 1e-5f);

  r4 = rotation3D({0, 0, -2}, a) * vec4(1, 2, 3, 1);
  EXPECT_EQ(r4.w, 1.0f);
  rotated = sub<3>(r4);
  EXPECT_VEC_NEAR(rotated, vec3(2, -1, 3), 1e-5f);

  r4 = rotation3D({3, 0, 0}, a) * vec4(1, 2, 3, 1);
  EXPECT_EQ(r4.w, 1.0f);
  rotated = sub<3>(r4);
  EXPECT_VEC_NEAR(rotated, vec3(1, -3, 2), 1e-5f);

  r4 = rotation3D({-4, 0, 0}, a) * vec4(1, 2, 3, 1);
  EXPECT_EQ(r4.w, 1.0f);
  rotated = sub<3>(r4);
  EXPECT_VEC_NEAR(rotated, vec3(1, 3, -2), 1e-5f);

  r4 = rotation3D({0, 5, 0}, a) * vec4(1, 2, 3, 1);
  EXPECT_EQ(r4.w, 1.0f);
  rotated = sub<3>(r4);
  EXPECT_VEC_NEAR(rotated, vec3(3, 2, -1), 1e-5f);

  r4 = rotation3D({0, -6, 0}, a) * vec4(1, 2, 3, 1);
  EXPECT_EQ(r4.w, 1.0f);
  rotated = sub<3>(r4);
  EXPECT_VEC_NEAR(rotated, vec3(-3, 2, 1), 1e-5f);
}

TEST(GeomTransform, Rotation3D_CubeDiag) {
  // Rotate around a diagonal of a unit cube starting at origin.
  vec3 axis(1, 1, 1);
  // A cube projected along the diagonal is a hexagon.
  // Rotating a cube vertex by 120 degrees (2*pi/3) should produce another vertex.
  mat3x4 rot = sub<3, 4>(rotation3D(axis, 2*M_PI/3));
  vec3 r;
  r = rot * vec4(1, 1, 0, 1);
  EXPECT_VEC_NEAR(r, vec3(0, 1, 1), 1e-5f);
  r = rot * vec4(0, 1, 1, 1);
  EXPECT_VEC_NEAR(r, vec3(1, 0, 1), 1e-5f);
  r = rot * vec4(1, 0, 1, 1);
  EXPECT_VEC_NEAR(r, vec3(1, 1, 0), 1e-5f);
}

TEST(GeomTransform, Shear2D) {
  auto shearx = shear(vec2(0.5f, 0));
  auto sheary = shear(vec2(0, 0.5f));
  EXPECT_EQ(shearx * vec3(1, 0, 1), vec3(1, 0, 1));
  EXPECT_EQ(shearx * vec3(0, 1, 1), vec3(0.5f, 1, 1));
  EXPECT_EQ(shearx * vec3(1, 1, 1), vec3(1.5f, 1, 1));

  EXPECT_EQ(sheary * vec3(1, 0, 1), vec3(1, 0.5f, 1));
  EXPECT_EQ(sheary * vec3(0, 1, 1), vec3(0, 1, 1));
  EXPECT_EQ(sheary * vec3(1, 1, 1), vec3(1, 1.5f, 1));
}

template <int n_out, int n_in, typename RNG>
void TestAffine(RNG &rng) {
  std::uniform_real_distribution<float> dist(-1, 1);
  mat<n_out, n_in + 1> m;
  for (int i = 0; i < m.rows; i++)
    for (int j = 0; j < m.cols; j++)
      m(i, j) = dist(rng);

  for (int iter = 0; iter < 100; iter++) {
    vec<n_in> v;
    for (int j = 0; j < n_in; j++)
      v[j] = dist(rng);

    vec<n_out> ref = m * cat(v, 1.0f);
    vec<n_out> out = affine(m, v);
    for (int i = 0; i < n_out; i++) {
      // the result may differ slightly due to different order of evaluation,
      // more suitable for contracting into FMA
      EXPECT_NEAR(ref[i], out[i], 1e-6);
    }
  }
}

TEST(GeomTransform, Affine) {
  std::mt19937_64 rng;
  TestAffine<2, 2>(rng);
  TestAffine<2, 2>(rng);
  TestAffine<3, 3>(rng);
  TestAffine<3, 4>(rng);
  TestAffine<4, 4>(rng);
}

}  // namespace dali
