// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>
#include <utility>
#include "dali/core/convert.h"
#include "dali/core/dev_buffer.h"
#include "dali/operators/random/rng_base_gpu.cuh"
#include "dali/operators/random/coin_flip.h"

namespace dali {

DALI_REGISTER_OPERATOR(random__CoinFlip, CoinFlip<GPUBackend>, GPU);
DALI_REGISTER_OPERATOR(CoinFlip, CoinFlip<GPUBackend>, GPU);  // deprecated alias

}  // namespace dali
