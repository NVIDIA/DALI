// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef DALI_PIPELINE_OPERATORS_UTIL_RANDOMIZER_IMPL_CPU_H_
#define DALI_PIPELINE_OPERATORS_UTIL_RANDOMIZER_IMPL_CPU_H_

#include "dali/pipeline/operators/util/randomizer.h"

namespace dali {

// CPU methods
template <>
Randomizer<CPUBackend>::Randomizer(int seed, size_t len) {}

template <>
int Randomizer<CPUBackend>::rand(int idx) {
  return lrand48();
}

template <>
void Randomizer<CPUBackend>::Cleanup() {}

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_UTIL_RANDOMIZER_IMPL_CPU_H_
