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
  explicit WarpAffineAugment(const OpSpec& spec) {}

  template <typename T>
  DISPLACEMENT_IMPL
  Point<T> operator()(int h, int w, int c, int H, int W, int C) {
    // TODO(ptredak): actual implementation
    Point<T> p;
    p.x = 0;
    p.y = 0;
    return p;
  }

  void Cleanup() {}
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
