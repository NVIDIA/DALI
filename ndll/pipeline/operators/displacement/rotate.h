// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifndef NDLL_PIPELINE_OPERATORS_DISPLACEMENT_ROTATE_H_
#define NDLL_PIPELINE_OPERATORS_DISPLACEMENT_ROTATE_H_

#include <vector>
#include <cmath>
#include "ndll/pipeline/operators/operator.h"
#include "ndll/pipeline/operators/displacement/warpaffine.h"

namespace ndll {

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

  virtual ~Rotate() = default;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_DISPLACEMENT_ROTATE_H_
