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


#ifndef DALI_PIPELINE_OPERATORS_DISPLACEMENT_ROTATE_H_
#define DALI_PIPELINE_OPERATORS_DISPLACEMENT_ROTATE_H_

#include <vector>
#include <cmath>
#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/displacement/warpaffine.h"

namespace dali {

class RotateAugment : public WarpAffineAugment {
 public:
  explicit RotateAugment(const OpSpec& spec) {
    use_image_center = true;
  }

  void Prepare(Param* p, const OpSpec& spec, ArgumentWorkspace *ws, int index) {
    float angle = spec.GetArgument<float>("angle", ws, index);
    float angle_rad = angle * M_PI / 180.0;
    p->matrix[0] = cos(angle_rad);
    p->matrix[1] = sin(angle_rad);
    p->matrix[2] = 0.0;
    p->matrix[3] = -sin(angle_rad);
    p->matrix[4] = cos(angle_rad);
    p->matrix[5] = 0.0;
  }
};

template <typename Backend>
class Rotate : public DisplacementFilter<Backend, RotateAugment> {
 public:
  inline explicit Rotate(const OpSpec &spec)
    : DisplacementFilter<Backend, RotateAugment>(spec) {}

  ~Rotate() override = default;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DISPLACEMENT_ROTATE_H_
