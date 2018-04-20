// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_DISPLACEMENT_FILTER_H_
#define NDLL_PIPELINE_OPERATORS_DISPLACEMENT_FILTER_H_

#include "ndll/common.h"
#include "ndll/pipeline/operator.h"

/**
 * @brief Provides a framework for doing displacement filter operations
 * such as flip, jitter, water, swirl, etc.
 */

namespace ndll {

class ColorIdentity {
 public:
  explicit ColorIdentity(const OpSpec& spec) {}

  template <typename T>
  __host__ __device__
  T operator()(const T in, const Index h, const Index w, const Index c,
               const Index H, const Index W, const Index C) {
    // identity
    return in;
  }

  void Cleanup() {}
};

class DisplacementIdentity {
 public:
  explicit DisplacementIdentity(const OpSpec& spec) {}

  __host__ __device__
  Index operator()(const Index h, const Index w, const Index c,
                   const Index H, const Index W, const Index C) {
    // identity
    return (h * W + w) * C + c;
  }

  void Cleanup() {}
};

template <typename Backend,
          class Displacement = DisplacementIdentity,
          class Augment = ColorIdentity,
          bool per_channel_transform = false>
class DisplacementFilter : public Operator<Backend> {};

}  // namespace ndll

#include "ndll/pipeline/operators/displacement_filter_impl_gpu.cuh"
#include "ndll/pipeline/operators/displacement_filter_impl_cpu.h"


#endif  // NDLL_PIPELINE_OPERATORS_DISPLACEMENT_FILTER_H_
