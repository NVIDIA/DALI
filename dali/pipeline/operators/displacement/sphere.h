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


#ifndef DALI_PIPELINE_OPERATORS_DISPLACEMENT_SPHERE_H_
#define DALI_PIPELINE_OPERATORS_DISPLACEMENT_SPHERE_H_

#include <ctgmath>
#include <vector>
#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/displacement/displacement_filter.h"

namespace dali {

class SphereAugment {
 public:
  explicit SphereAugment(const OpSpec& spec) {}

  template <typename T>
  DISPLACEMENT_IMPL
  Point<T> operator()(int h, int w, int c, int H, int W, int C) {
    // SPHERE_PREAMBLE
    const int mid_x = W / 2;
    const int mid_y = H / 2;
    const int d = mid_x > mid_y ? mid_x : mid_y;

    // SPHERE_CORE
    const int trueY = h - mid_y;
    const int trueX = w - mid_x;
    const float rad = sqrtf(trueX * trueX + trueY * trueY) / d;

    T newX = mid_x + rad * trueX;
    T newY = mid_y + rad * trueY;

    return CreatePointLimited(newX, newY, W, H);
  }

  void Cleanup() {}
};

template <typename Backend>
class Sphere : public DisplacementFilter<Backend, SphereAugment> {
 public:
    inline explicit Sphere(const OpSpec &spec)
      : DisplacementFilter<Backend, SphereAugment>(spec) {}


    ~Sphere() override = default;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DISPLACEMENT_SPHERE_H_
