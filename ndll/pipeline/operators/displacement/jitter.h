// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifndef NDLL_PIPELINE_OPERATORS_DISPLACEMENT_JITTER_H_
#define NDLL_PIPELINE_OPERATORS_DISPLACEMENT_JITTER_H_

#include <ctgmath>
#include <vector>
#include "ndll/pipeline/operators/operator.h"
#include "ndll/pipeline/operators/displacement/displacement_filter.h"
#include "ndll/pipeline/operators/util/randomizer.h"

namespace ndll {

template <typename Backend>
class JitterAugment {
 public:
  explicit JitterAugment(const OpSpec& spec) :
        nDegree_(spec.GetArgument<int>("nDegree")),
        rnd_(spec.GetArgument<int>("seed"), 128*256) {}

template <typename T>
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  Point<T> operator()(int y, int x, int c, int H, int W, int C) {
    // JITTER_PREAMBLE
    const uint16_t degr = nDegree_;
    const uint16_t nHalf = degr/2;


    // JITTER_CORE
#ifdef __CUDA_ARCH__
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
#else
    const int idx = 0;
#endif

    Point<T> p;
    const int newX = rnd_.rand(idx) % degr - nHalf + x;
    const int newY = rnd_.rand(idx) % degr - nHalf + y;

    p.x = newX >= 0 && newX < W ? newX : -1;
    p.y = newY >= 0 && newY < H ? newY : -1;

    return p;
  }

  void Cleanup() {
    rnd_.Cleanup();
  }

 private:
  const size_t nDegree_;
  Randomizer<Backend> rnd_;
};

template <typename Backend>
class Jitter : public DisplacementFilter<Backend, JitterAugment<Backend>> {
 public:
    inline explicit Jitter(const OpSpec &spec)
      : DisplacementFilter<Backend, JitterAugment<Backend>>(spec) {}

    virtual ~Jitter() = default;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_DISPLACEMENT_JITTER_H_
