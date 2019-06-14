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

#ifndef DALI_KERNELS_IMGPROC_WARP_AFFINE_H_
#define DALI_KERNELS_IMGPROC_WARP_AFFINE_H_

#include "dali/core/geom/mat.h"

namespace dali {
namespace kernels {

template <int dim>
struct AffineMapping {
  mat<dim, dim+1> transform;
  AffineMapping() = default;
  DALI_HOST_DEV
  constexpr AffineMapping(const mat<dim, dim+1> &m) : transform(m) {}  // NOLINT

  DALI_HOST_DEV
  vec<dim> operator()(const vec<dim> &v) {
    // return transform*cat(v, 1.0f);  // this wastes `dim` multiplications
    return transform.col(dim-1) + sub<dim, dim>(transform) * v;
  }
};

using AffineMapping2D = AffineMapping<2>;
using AffineMapping3D = AffineMapping<3>;

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_WARP_AFFINE_H_
