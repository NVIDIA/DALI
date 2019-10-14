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

#ifndef DALI_OPERATORS_UTIL_RANDOMIZER_IMPL_CPU_H_
#define DALI_OPERATORS_UTIL_RANDOMIZER_IMPL_CPU_H_

#include "dali/operators/util/randomizer.h"
#if !defined(__AARCH64_QNX__)
#include <stdlib.h>
#else
#include <random>
#include <limits>
#endif

namespace dali {

// CPU methods
template <>
Randomizer<CPUBackend>::Randomizer(int seed, size_t len) {}

template <>
int Randomizer<CPUBackend>::rand(int idx) {
#if !defined(__AARCH64_QNX__)
  return lrand48();
#else
  // TODO(klecki): Use QNX lrand48
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<> dist(0, std::numeric_limits<int>::max());
  return dist(mt);
#endif
}

template <>
void Randomizer<CPUBackend>::Cleanup() {}

}  // namespace dali

#endif  // DALI_OPERATORS_UTIL_RANDOMIZER_IMPL_CPU_H_
