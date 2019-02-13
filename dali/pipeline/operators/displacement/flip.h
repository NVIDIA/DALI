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


#ifndef DALI_PIPELINE_OPERATORS_DISPLACEMENT_FLIP_H_
#define DALI_PIPELINE_OPERATORS_DISPLACEMENT_FLIP_H_

#include <vector>
#include <cmath>
#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/displacement/warpaffine.h"

namespace dali {

class FlipAugment : public WarpAffineAugment {
 public:
  explicit FlipAugment(const OpSpec& spec) {
    use_image_center = true;
  }

  void Prepare(Param* p, const OpSpec& spec, ArgumentWorkspace *ws, int index) {
    bool horizontal = 0 != spec.GetArgument<int>("horizontal", ws, index);
    bool vertical   = 0 != spec.GetArgument<int>("vertical", ws, index);
    // NOTE:
    // There are additional -1px offsets for flipped axes because use_image_center is broken.
    // To be removed when displacement is done properly.
    p->matrix[0] = horizontal ? -1 : 1;
    p->matrix[1] = 0;
    p->matrix[2] = horizontal ? -1 : 0;
    p->matrix[3] = 0;
    p->matrix[4] = vertical ? -1 : 1;
    p->matrix[5] = vertical ? -1 : 0;
  }
};

template <typename Backend>
class Flip : public DisplacementFilter<Backend, FlipAugment> {
 public:
  inline explicit Flip(const OpSpec &spec)
    : DisplacementFilter<Backend, FlipAugment>(spec) {}

  ~Flip() override = default;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DISPLACEMENT_FLIP_H_
