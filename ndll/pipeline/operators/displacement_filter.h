// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_DISPLACEMENT_FILTER_H_
#define NDLL_PIPELINE_OPERATORS_DISPLACEMENT_FILTER_H_

#include "ndll/common.h"
#include "ndll/pipeline/operator.h"

/**
 * @brief Provides a framework for doing displacement filter operations
 * such as flip, jitter, water, swirl, etc.
 */

#ifndef DISPLACEMENT_IMPL
#ifdef __CUDA_ARCH__
#define DISPLACEMENT_IMPL __host__ __device__
#else
#define DISPLACEMENT_IMPL
#endif
#endif

#define DISPLACEMENT_SCHEMA_ARGS \
  .AddOptionalArg("probability", \
      "Probability of applying this augmentation to the input image", 1.0f) \
  .AddOptionalArg("interp_type", "Type of interpolation used", NDLL_INTERP_NN)

namespace ndll {

template <typename T>
struct Point {
  T x, y;
};

class DisplacementIdentity {
 public:
  explicit DisplacementIdentity(const OpSpec& spec) {}

  template<typename T>
  DISPLACEMENT_IMPL
  Point<T> operator()(const Index h, const Index w, const Index c,
                      const Index H, const Index W, const Index C) {
    // identity
    Point<T> p;
    p.x = w;
    p.y = h;
    return p;
  }

  void Cleanup() {}
};

template <typename Backend,
          class Displacement = DisplacementIdentity,
          bool per_channel_transform = false>
class DisplacementFilter : public Operator<Backend> {};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_DISPLACEMENT_FILTER_H_
