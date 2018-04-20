// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifndef NDLL_PIPELINE_OPERATORS_RANDOMIZER_H_
#define NDLL_PIPELINE_OPERATORS_RANDOMIZER_H_

#include "ndll/pipeline/data/backend.h"

namespace ndll {

template <typename Backend>
class Randomizer {
 public:
  explicit Randomizer(int seed = 1234, size_t len = 128*32*32);

  int rand(int idx);

  void Cleanup();

 private:
    void *states_;
    size_t len_;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_RANDOMIZER_H_
