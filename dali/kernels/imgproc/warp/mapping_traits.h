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

#ifndef DALI_KERNELS_IMGPROC_WARP_MAPPING_TRAITS_H_
#define DALI_KERNELS_IMGPROC_WARP_MAPPING_TRAITS_H_

#include <functional>
#include "dali/core/geom/vec.h"

namespace dali {
namespace kernels {
namespace warp {

std::true_type is_fp_mapping_helper(const std::function<vec2(vec2)> &f);
std::true_type is_fp_mapping_helper(const std::function<vec3(vec3)> &f);
std::true_type is_fp_mapping_helper(const std::function<vec4(vec4)> &f);

std::false_type is_fp_mapping_helper(const std::function<ivec2(ivec2)> &f);
std::false_type is_fp_mapping_helper(const std::function<ivec3(ivec3)> &f);
std::false_type is_fp_mapping_helper(const std::function<ivec4(ivec4)> &f);


template <typename Mapping>
struct is_fp_mapping : decltype(is_fp_mapping_helper(Mapping())) {};

template <typename Mapping>
struct mapping_params {
  using type = Mapping;
};

/**
 * @brief This type is passed to the Warp kernel to construct the mapping object
 *
 * The Mapping object can be transient and/or contain additional state.
 * mapping_params_t type is used to distinguish between mapping object and
 * the parameters needed to construct one - by default they are the same type.
 */
template <typename Mapping>
using mapping_params_t = typename mapping_params<Mapping>::type;

}  // namespace warp
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_WARP_MAPPING_TRAITS_H_
