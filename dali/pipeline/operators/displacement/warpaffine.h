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

  template <typename T>
  DISPLACEMENT_IMPL
  Point<T> operator()(int h, int w, int c, int H, int W, int C) {
    T hp = h;
    T wp = w;
    if (use_image_center) {
      hp -= H/2.0f;
      wp -= W/2.0f;
    }
    T newX = param.matrix[0] * wp + param.matrix[1] * hp + param.matrix[2];
    T newY = param.matrix[3] * wp + param.matrix[4] * hp + param.matrix[5];
    if (use_image_center) {
      newX += W/2.0f;
      newY += H/2.0f;
    }

    return CreatePointLimited(newX, newY, W, H);
  }

  void Cleanup() {}

  struct Param {
    float matrix[6];
  };

  Param param;

  void Prepare(Param* p, const OpSpec& spec, ArgumentWorkspace *ws, int index) {
    std::vector<float> tmp;
    GetSingleOrRepeatedArg(spec, &tmp, "matrix", size);
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
