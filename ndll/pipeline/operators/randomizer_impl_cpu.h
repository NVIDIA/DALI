// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_RANDOMIZER_IMPL_CPU_H_
#define NDLL_PIPELINE_OPERATORS_RANDOMIZER_IMPL_CPU_H_

#include "ndll/pipeline/operators/randomizer.h"

namespace ndll {

// CPU methods
template <>
Randomizer<CPUBackend>::Randomizer(int seed, size_t len) {}

template <>
int Randomizer<CPUBackend>::rand(int idx) {
  return lrand48();
}

template <>
void Randomizer<CPUBackend>::Cleanup() {}

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_RANDOMIZER_IMPL_CPU_H_
