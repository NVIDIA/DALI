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

#ifndef DALI_KERNELS_IMGPROC_WARP_MAP_COORDS_H_
#define DALI_KERNELS_IMGPROC_WARP_MAP_COORDS_H_

#include "dali/kernels/kernel.h"
#include "dali/kernels/imgproc/sampler.h"
#include "dali/kernels/imgproc/warp/warp_setup.cuh"
#include "dali/kernels/imgproc/warp/mapping_traits.h"

namespace dali {
namespace kernels {
namespace warp {

/** @brief Converts destination _integer_ coordinates to _floating point_ source coordinates
 *
 * This function is used by warp implementations to calculate source coordinates using
 * using non-integer mapping. The coordinates are converted to pixel-centered.
 */
template <typename Mapping, int dim>
DALI_HOST_DEV
inline enable_if_t<is_fp_mapping<Mapping>::value, vec<dim>>
map_coords(const Mapping &m, ivec<dim> pos) {
  // When given floating point coordinates, samplers expect pixel centers to be offset by half.
  // Since ultimately our destination coordinates are integer, we need to move to pixel-center
  // frame before applying the mapping and passing the output to a sampler.
  return m(pos + 0.5f);
}

/** @brief Converts destination _integer_ coordinates to _integer_ source coordinates
 *
 * This function is used by warp implementations to calculate source coordinates using
 * using integer mapping.
 */
template <typename Mapping, int dim>
DALI_HOST_DEV
inline enable_if_t<!is_fp_mapping<Mapping>::value, ivec<dim>>
map_coords(const Mapping &m, ivec<dim> pos) {
  // When given integer point coordinates, samplers simply uses them as 0-based indices.
  // If the mapping producces integral coordinates, there's no point in going from indices to
  // pixel centers and then back to integral coordinates - hence, 0.5 is not added.
  return m(pos);
}

}  // namespace warp
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_WARP_MAP_COORDS_H_
