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

#ifndef DALI_PIPELINE_OPERATORS_UTIL_RANDOMIZER_IMPL_CPU_H_
#define DALI_PIPELINE_OPERATORS_UTIL_RANDOMIZER_IMPL_CPU_H_

#include "dali/pipeline/operators/util/randomizer.h"
#include <stdlib.h>

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
