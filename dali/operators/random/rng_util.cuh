// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_RANDOM_RNG_UTIL_CUH_
#define DALI_OPERATORS_RANDOM_RNG_UTIL_CUH_

#include "dali/core/host_dev.h"
#include "dali/core/force_inline.h"
#include "dali/core/int_literals.h"
#include "dali/operators/random/philox.h"
#include <curand_kernel.h>  // NOLINT

namespace dali {

__device__ DALI_FORCEINLINE curandStatePhilox4_32_10_t ToCurand(Philox4x32_10::State state) {
  curandStatePhilox4_32_10_t rng{};
  curand_init(state.key, state.ctr[1], state.ctr[0] << 2 | state.phase, &rng);
  // the two high bits of the counter cannot be set in init, so we need this loop
  #pragma unroll 4
  for (unsigned i = 0; i < state.ctr[0] >> 62; i++) {
    skipahead(1_u64 << 63, &rng);
  }
  return rng;
}

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_RNG_UTIL_CUH_
