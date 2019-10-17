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

#ifndef DALI_CORE_GEOM_TRANSFORM_H_
#define DALI_CORE_GEOM_TRANSFORM_H_

#include "dali/core/geom/mat.h"

namespace dali {

DALI_HOST_DEV
constexpr mat3 translation(vec2 offset) {
  return {{
    { 1, 0, offset.x },
    { 0, 1, offset.y },
    { 0, 0, 1 }
  }};
}

DALI_HOST_DEV
constexpr mat4 translation(vec3 offset) {
  return {{
    { 1, 0, 0, offset.x },
    { 0, 1, 0, offset.y },
    { 0, 0, 1, offset.z },
    { 0, 0, 0, 1 }
  }};
}

DALI_HOST_DEV
constexpr mat3 scaling(vec2 scale) {
  return {{
    { scale.x, 0, 0 },
    { 0, scale.y, 0 },
    { 0, 0,       1 }
  }};
}

DALI_HOST_DEV
constexpr mat4 scaling(vec3 scale) {
  return {{
    { scale.x, 0, 0, 0 },
    { 0, scale.y, 0, 0 },
    { 0, 0, scale.z, 0 },
    { 0, 0, 0,       1 }
  }};
}

DALI_HOST_DEV
inline mat3 rotation2D(float angle) {
#ifdef __CUDA_ARCH__
  float c = cosf(angle);
  float s = sinf(angle);
#else
  float c = std::cos(angle);
  float s = std::sin(angle);
#endif
  return {{
    { c, -s, 0 },
    { s, c, 0 },
    { 0, 0, 1 }
  }};
}

DALI_HOST_DEV
constexpr mat3 shear(vec2 shear) {
  return {{
    { 1, shear.x, 0 },
    { shear.y, 1, 0 },
    { 0, 0, 1 }
  }};
}

DALI_HOST_DEV
constexpr mat4 shear(mat3x2 shear) {
  return {{
    { 1,           shear(0, 0), shear(0, 1), 0 },
    { shear(1, 0), 1,           shear(1, 1), 0 },
    { shear(2, 0), shear(2, 1), 1,           0 },
    { 0,           0,           0,           1 }
  }};
}

template <int out_n, int in_n>
DALI_HOST_DEV
constexpr vec<out_n> affine(const mat<out_n, in_n + 1> &transform, const vec<in_n> &v) {
  vec<out_n> out = {};
  for (int i = 0; i < out_n; i++) {
    // NOTE: accumulating directly in out[i] prodced noticeably slower code in GCC 7.4
    float sum = transform(i, in_n);
    for (int j = 0; j < in_n; j++) {
      sum += transform(i, j) * v[j];
    }
    out[i] = sum;
  }
  return out;
}

template<>
DALI_HOST_DEV
constexpr vec2 affine<2, 2>(const mat<2, 3> &transform, const vec2 &v) {
  return {
    transform(0, 2) + transform(0, 0) * v.x + transform(0, 1) * v.y,
    transform(1, 2) + transform(1, 0) * v.x + transform(1, 1) * v.y,
  };
}

template<>
DALI_HOST_DEV
constexpr vec3 affine<3, 3>(const mat<3, 4> &transform, const vec3 &v) {
  return {
    transform(0, 3) + transform(0, 0) * v.x + transform(0, 1) * v.y + transform(0, 2) * v.z,
    transform(1, 3) + transform(1, 0) * v.x + transform(1, 1) * v.y + transform(1, 2) * v.z,
    transform(2, 3) + transform(2, 0) * v.x + transform(2, 1) * v.y + transform(2, 2) * v.z,
  };
}

}  // namespace dali

#endif  // DALI_CORE_GEOM_TRANSFORM_H_
