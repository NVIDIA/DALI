// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

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
