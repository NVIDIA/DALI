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

}  // namespace dali
