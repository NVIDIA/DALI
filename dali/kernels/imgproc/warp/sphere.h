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

#ifndef DALI_KERNELS_IMGPROC_WARP_SPHERE_H_
#define DALI_KERNELS_IMGPROC_WARP_SPHERE_H_

#include "dali/core/geom/vec.h"

namespace dali {
namespace kernels {

struct SphereMapping {
  static SphereMapping FromSizes(ivec2 out_size, ivec2 in_size) {
    SphereMapping m;
    m.src_center = in_size * 0.5f;
    m.dst_center = out_size * 0.5f;
    m.src_scale = in_size.cast<float>() / out_size;
    m.dst_scale = 1.0f / std::min(m.src_center.x, m.src_center.y);
    return m;
  }

  vec2 src_center, dst_center, src_scale, dst_scale;

  DALI_HOST_DEV
  vec2 operator()(vec2 dst_pos) const {
    dst_pos -= dst_center;
    return dst_pos * src_scale * (dst_pos * dst_scale).length() + src_center;
  }
};

}  // namespace kernels
}  // namespace dali


#endif  // DALI_KERNELS_IMGPROC_WARP_SPHERE_H_
