// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifndef DALI_PIPELINE_OPERATORS_UTIL_RANDOMIZER_H_
#define DALI_PIPELINE_OPERATORS_UTIL_RANDOMIZER_H_

#include "dali/pipeline/data/backend.h"

namespace dali {

template <typename Backend>
class Randomizer {
 public:
  explicit Randomizer(int seed = 1234, size_t len = 128*32*32);

#if __CUDA_ARCH__
  __device__
#endif
  int rand(int idx);

  void Cleanup();

 private:
    void *states_;
    size_t len_;
    int device_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_UTIL_RANDOMIZER_H_
