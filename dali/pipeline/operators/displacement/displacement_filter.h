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

#ifndef DALI_PIPELINE_OPERATORS_DISPLACEMENT_DISPLACEMENT_FILTER_H_
#define DALI_PIPELINE_OPERATORS_DISPLACEMENT_DISPLACEMENT_FILTER_H_

#include "dali/core/common.h"
#include "dali/core/host_dev.h"
#include "dali/pipeline/operators/operator.h"

/**
 * @brief Provides a framework for doing displacement filter operations
 * such as flip, jitter, water, swirl, etc.
 */

namespace dali {

template <typename T, typename = int>
struct HasParam : std::false_type { };

template <typename T>
struct HasParam <T, decltype((void) (typename T::Param()), 0)> : std::true_type {};

template <typename T>
struct Point {
  const T x, y;
  Point() = delete;

  template <typename U>
  Point<U> Cast() {
    return {static_cast<U>(x), static_cast<U>(y)};
  }
};

template <typename T>
DALI_HOST_DEV
T ToValidCoord(T coord, Index limit) {
  return coord >= 0 && coord < limit ? coord : -1;
}
template <typename T>
DALI_HOST_DEV
Point<T> CreatePointLimited(T x, T y, Index W, Index H) {
  return {ToValidCoord(x, W), ToValidCoord(y, H)};
}

class DisplacementIdentity {
 public:
  explicit DisplacementIdentity(const OpSpec& spec) {}

  DALI_HOST_DEV
  Point<int> operator()(const int h, const int w, const int c,
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

#endif  // DALI_PIPELINE_OPERATORS_DISPLACEMENT_DISPLACEMENT_FILTER_H_
