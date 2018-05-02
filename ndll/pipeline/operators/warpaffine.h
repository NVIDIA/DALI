// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifndef NDLL_PIPELINE_OPERATORS_WARPAFFINE_H_
#define NDLL_PIPELINE_OPERATORS_WARPAFFINE_H_

#include <ctgmath>
#include <vector>
#include "ndll/pipeline/operator.h"
#include "ndll/pipeline/operators/displacement_filter.h"

namespace ndll {

class WarpAffineAugment {
 public:
  static const int size = 6;
  explicit WarpAffineAugment(const OpSpec& spec) {
    std::vector<float> tmp = spec.GetRepeatedArgument<float>("matrix");
    NDLL_ENFORCE(tmp.size() == size, "Warp affine matrix needs to have 6 elements");
    for (int i = 0; i < size; ++i) {
      matrix[i] = tmp[i];
    }
  }

  template <typename T>
  DISPLACEMENT_IMPL
  Point<T> operator()(int h, int w, int c, int H, int W, int C) {
    // TODO(ptredak): actual implementation
    Point<T> p;
    const T newX = matrix[0] * w + matrix[1] * h + matrix[2];
    const T newY = matrix[3] * w + matrix[4] * h + matrix[5];
    p.x = newX > 0 && newX < W ? newX : 0;
    p.y = newY > 0 && newY < H ? newY : 0;
    return p;
  }

  void Cleanup() {}

 private:
  float matrix[6];
};

template <typename Backend>
class WarpAffine : public DisplacementFilter<Backend, WarpAffineAugment> {
 public:
    inline explicit WarpAffine(const OpSpec &spec)
      : DisplacementFilter<Backend, WarpAffineAugment>(spec) {}


    virtual ~WarpAffine() = default;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_WARPAFFINE_H_
