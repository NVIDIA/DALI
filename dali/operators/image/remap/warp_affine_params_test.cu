// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include "dali/operators/image/remap/warp_affine_params.h"
#include "dali/core/cuda_error.h"

namespace dali {

// Regression test for https://github.com/NVIDIA/DALI/issues/6504
// A zero-sample batch used to cause a host-side division by zero while computing the CUDA
// launch grid (`div_ceil(count, 512)` / `std::min(count, 512)` with `count == 0` is fine,
// but the original bug was in a sibling launcher with the same shape; this guards the
// `count == 0` early return added to `InvertTransforms`/`CopyTransforms`).
TEST(WarpAffineParamsTest, CopyTransformsGPUDoesNotCrashOnZeroCount) {
  ASSERT_NO_THROW(
      (CopyTransformsGPU<2, false>(nullptr, nullptr, 0, 0)));
  ASSERT_NO_THROW(
      (CopyTransformsGPU<2, true>(nullptr, nullptr, 0, 0)));
  ASSERT_NO_THROW(
      (CopyTransformsGPU<3, false>(nullptr, nullptr, 0, 0)));
  ASSERT_NO_THROW(
      (CopyTransformsGPU<3, true>(nullptr, nullptr, 0, 0)));
  CUDA_CALL(cudaStreamSynchronize(0));
}

}  // namespace dali
