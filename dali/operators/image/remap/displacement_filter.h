// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_REMAP_DISPLACEMENT_FILTER_H_
#define DALI_OPERATORS_IMAGE_REMAP_DISPLACEMENT_FILTER_H_

#include "dali/core/common.h"
#include "dali/core/geom/vec.h"
#include "dali/pipeline/operator/operator.h"

/**
 * @brief Provides a framework for doing displacement filter operations
 * such as flip, jitter, water, swirl, etc.
 */

namespace dali {

template <typename T, typename = int>
struct HasParam : std::false_type { };

template <typename T>
struct HasParam <T, decltype((void) (typename T::Param()), 0)> : std::true_type {};

class DisplacementIdentity {
 public:
  explicit DisplacementIdentity(const OpSpec& spec) {}

  DALI_HOST_DEV
  ivec2 operator()(const int h, const int w, const int c,
                   const int H, const int W, const int C) {
    // identity
    return { w, h };
  }

  void Cleanup() {}
};

template <typename Backend,
          class Displacement = DisplacementIdentity,
          bool per_channel_transform = false>
class DisplacementFilter : public Operator<Backend> {};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_REMAP_DISPLACEMENT_FILTER_H_
