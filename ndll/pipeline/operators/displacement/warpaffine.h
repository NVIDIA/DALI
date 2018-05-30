// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifndef NDLL_PIPELINE_OPERATORS_DISPLACEMENT_WARPAFFINE_H_
#define NDLL_PIPELINE_OPERATORS_DISPLACEMENT_WARPAFFINE_H_

#include <vector>
#include "ndll/pipeline/operators/operator.h"
#include "ndll/pipeline/operators/displacement/displacement_filter.h"

namespace ndll {

class WarpAffineAugment {
 public:
  static const int size = 6;

  WarpAffineAugment() {}

  explicit WarpAffineAugment(const OpSpec& spec)
    : use_image_center(spec.GetArgument<bool>("use_image_center")) {}

  template <typename T>
  DISPLACEMENT_IMPL
  Point<T> operator()(int h, int w, int c, int H, int W, int C) {
    Point<T> p;
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

    p.x = newX >= 0 && newX < W ? newX : -1;
    p.y = newY >= 0 && newY < H ? newY : -1;
    return p;
  }

  void Cleanup() {}

  struct Param {
    float matrix[6];
  };

  Param param;

  void Prepare(Param* p, const OpSpec& spec, ArgumentWorkspace *ws, int index) {
    const std::vector<float>& tmp = spec.GetRepeatedArgument<float>("matrix");
    NDLL_ENFORCE(tmp.size() == size, "Warp affine matrix needs to have 6 elements");
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

    virtual ~WarpAffine() = default;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_DISPLACEMENT_WARPAFFINE_H_
