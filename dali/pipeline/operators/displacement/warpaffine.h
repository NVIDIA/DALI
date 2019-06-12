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


#ifndef DALI_PIPELINE_OPERATORS_DISPLACEMENT_WARPAFFINE_H_
#define DALI_PIPELINE_OPERATORS_DISPLACEMENT_WARPAFFINE_H_

#include <vector>
#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/displacement/displacement_filter.h"
#include "dali/pipeline/operators/common.h"

namespace dali {

class WarpAffineAugment {
 public:
  static const int size = 6;

  WarpAffineAugment() {}

  explicit WarpAffineAugment(const OpSpec& spec)
    : use_image_center(spec.GetArgument<bool>("use_image_center")) {}

  DALI_HOST_DEV
  Point<float> operator()(int iy, int ix, int c, int H, int W, int C) {
    float y = iy + 0.5f;
    float x = ix + 0.5f;
    if (use_image_center) {
      y -= H*0.5f;
      x -= W*0.5f;
    }
    float newX = param.matrix[0] * x + param.matrix[1] * y + param.matrix[2];
    float newY = param.matrix[3] * x + param.matrix[4] * y + param.matrix[5];
    if (use_image_center) {
      newX += W*0.5f;
      newY += H*0.5f;
    }

    return { newX, newY };
  }

  void Cleanup() {}

  struct Param {
    float matrix[6];
  };

  Param param;

  void Prepare(Param* p, const OpSpec& spec, ArgumentWorkspace *ws, int index) {
    std::vector<float> tmp;
    GetSingleOrRepeatedArg(spec, tmp, "matrix", size);
    for (int i = 0; i < size; ++i) {
      p->matrix[i] = tmp[i];
    }
  }

 protected:
  bool use_image_center;
};

template <typename Backend>
class WarpAffine : public DisplacementFilter<Backend, WarpAffineAugment> {
 public:
    inline explicit WarpAffine(const OpSpec &spec)
      : DisplacementFilter<Backend, WarpAffineAugment>(spec) {}

    ~WarpAffine() override = default;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DISPLACEMENT_WARPAFFINE_H_
