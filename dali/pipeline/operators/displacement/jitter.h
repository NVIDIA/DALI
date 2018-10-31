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


#ifndef DALI_PIPELINE_OPERATORS_DISPLACEMENT_JITTER_H_
#define DALI_PIPELINE_OPERATORS_DISPLACEMENT_JITTER_H_

#include <ctgmath>
#include <vector>
#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/displacement/displacement_filter.h"
#include "dali/pipeline/operators/util/randomizer.h"

namespace dali {

template <typename Backend>
class JitterAugment {
 public:
  explicit JitterAugment(const OpSpec& spec) :
        nDegree_(spec.GetArgument<int>("nDegree")),
        rnd_(spec.GetArgument<int64_t>("seed"), 128*256) {}

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

    const T newX = rnd_.rand(idx) % degr - nHalf + x;
    const T newY = rnd_.rand(idx) % degr - nHalf + y;

    return CreatePointLimited(newX, newY, W, H);
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

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DISPLACEMENT_JITTER_H_
