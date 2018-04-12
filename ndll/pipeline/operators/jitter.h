// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifndef NDLL_PIPELINE_OPERATORS_JITTER_H_
#define NDLL_PIPELINE_OPERATORS_JITTER_H_

#include <ctgmath>
#include <vector>
#include "ndll/pipeline/operator.h"
#include "ndll/pipeline/operators/displacement_filter.h"
#include "ndll/pipeline/operators/randomizer.h"

namespace ndll {

class JitterAugment {
 public:
  explicit JitterAugment(const OpSpec& spec) :
        nDegree_(spec.GetArgument<int>("nDegree")),
        rnd_(spec.GetArgument<int>("seed"), 128*256) {}

  __host__ __device__
  Index operator()(int y, int x, int c, int H, int W, int C) {
    // JITTER_PREAMBLE
    const uint16_t degr = nDegree_;
    const uint16_t nHalf = degr/2;
    const int nYoffset = (W * C + 1) / 2 * 2;


    // JITTER_CORE
    int newX, newY;
#ifdef __CUDA_ARCH__
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
#else
    const int idx = 0;
#endif

    const uint32_t from = (newX = rnd_.rand(idx) % degr - nHalf + x) > 0 && newX < W && \
                          (newY = rnd_.rand(idx) % degr - nHalf + y) > 0 && newY < H?   \
                          newX * C + newY * nYoffset + c : (y * W + x) * C + c;
    return from;
  }

  void Cleanup() {
    rnd_.Cleanup();
  }

 private:
  Randomizer rnd_;
  const size_t nDegree_;
};

template <typename Backend>
class Jitter : public DisplacementFilter<Backend, JitterAugment, ColorIdentity> {
 public:
    inline explicit Jitter(const OpSpec &spec)
      : DisplacementFilter<Backend, JitterAugment>(spec) {}

    virtual ~Jitter() = default;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_JITTER_H_
