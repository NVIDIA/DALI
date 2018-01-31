// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifndef NDLL_PIPELINE_OPERATORS_SPHERE_NEW_H_
#define NDLL_PIPELINE_OPERATORS_SPHERE_NEW_H_

#include <ctgmath>
#include <vector>
#include "ndll/pipeline/operator.h"
#include "ndll/pipeline/operators/displacement_filter.h"

namespace ndll {

class SphereAugment {
 public:
  explicit SphereAugment(const OpSpec& spec) {}

  __host__ __device__
  Index operator()(int h, int w, int c, int H, int W, int C) {
    // SPHERE_PREAMBLE
    const int mid_x = W / 2;
    const int mid_y = H / 2;
    const int d = mid_x > mid_y ? mid_x : mid_y;
    const int nYoffset = (W * C + 1) / 2 * 2;

    // SPHERE_CORE
    const int trueY = h - mid_y;
    const int trueX = w - mid_x;
    const float rad = sqrtf(trueX * trueX + trueY * trueY) / d;

    int newX = mid_x + rad * trueX;
    int newY = mid_y + rad * trueY;

    const int from_idx = newX > 0 && newX < W && newY > 0 && newY < H ?
      newX * C + newY * nYoffset + c : c;

    return from_idx;
  }

  void Cleanup() {}
};

template <typename Backend>
class SphereNew : public DisplacementFilter<Backend, SphereAugment, ColorIdentity> {
 public:
    inline explicit SphereNew(const OpSpec &spec)
      : DisplacementFilter<Backend, SphereAugment, ColorIdentity>(spec) {}


    virtual ~SphereNew() = default;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_SPHERE_NEW_H_
