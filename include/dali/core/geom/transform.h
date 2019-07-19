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
constexpr mat3 rotation2D(float angle) {
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

}  // namespace dali

#endif  // DALI_CORE_GEOM_TRANSFORM_H_
