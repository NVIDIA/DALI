// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_COLOR_SPACE_CONVERSION_IMPL_H_
#define DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_COLOR_SPACE_CONVERSION_IMPL_H_

#include <cuda_runtime_api.h>
#include "dali/core/geom/vec.h"
#include "dali/core/convert.h"

namespace dali {
namespace kernels {

namespace {

template <typename Input>
DALI_HOST_DEV DALI_FORCEINLINE vec<3, float> norm(vec<3, Input> rgb) {
  return vec<3, float>{ConvertNorm<float>(rgb.x), ConvertNorm<float>(rgb.y),
                       ConvertNorm<float>(rgb.z)};
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE vec<3, float> norm(vec<3, float> rgb) {
  return rgb;
}

}  // namespace

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE Output rgb_to_y(vec<3, Input> rgb_in) {
  auto rgb = norm(rgb_in);
  return ConvertNorm<Output>(0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z);
}

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE Output rgb_to_cb(vec<3, Input> rgb_in) {
  auto rgb = norm(rgb_in);
  return ConvertNorm<Output>(-0.16873589f * rgb.x - 0.33126411f * rgb.y + 0.50000000f * rgb.z +
                             0.5f);
}

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE Output rgb_to_cr(vec<3, Input> rgb_in) {
  auto rgb = norm(rgb_in);
  return ConvertNorm<Output>(0.50000000f * rgb.x - 0.41868759f * rgb.y - 0.08131241f * rgb.z +
                             0.5f);
}

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE vec<2, Output> rgb_to_cb_cr(vec<3, Input> rgb_in) {
  auto rgb = norm(rgb_in);
  return {rgb_to_cb<Output>(rgb), rgb_to_cr<Output>(rgb)};
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_COLOR_SPACE_CONVERSION_IMPL_H_
