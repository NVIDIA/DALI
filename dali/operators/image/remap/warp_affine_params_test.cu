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
#include "dali/core/mm/memory.h"

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

// Companion test that exercises the same launchers with a non-empty batch and checks the
// actual output against a reference value, so the zero-count guard above can't be
// satisfied by a launcher that silently does nothing for every input.
TEST(WarpAffineParamsTest, CopyTransformsGPUCopiesAndInvertsSingleSample) {
  using Params = WarpAffineParams<2>;
  // Pure translation: trivial to invert exactly by hand (negate the translation column).
  Params transform{mat<2, 3>{{{1, 0, 5}, {0, 1, 3}}}};
  Params expected_inv{mat<2, 3>{{{1, 0, -5}, {0, 1, -3}}}};

  auto d_input = mm::alloc_raw_unique<Params, mm::memory_kind::device>(1);
  auto d_output = mm::alloc_raw_unique<Params, mm::memory_kind::device>(1);
  auto d_input_ptrs = mm::alloc_raw_unique<const Params *, mm::memory_kind::device>(1);
  CUDA_CALL(cudaMemcpy(d_input.get(), &transform, sizeof(Params), cudaMemcpyHostToDevice));
  const Params *h_input_ptr = d_input.get();
  CUDA_CALL(
      cudaMemcpy(d_input_ptrs.get(), &h_input_ptr, sizeof(Params *), cudaMemcpyHostToDevice));

  CopyTransformsGPU<2, false>(d_output.get(), d_input_ptrs.get(), 1, 0);
  CUDA_CALL(cudaStreamSynchronize(0));
  Params copied;
  CUDA_CALL(cudaMemcpy(&copied, d_output.get(), sizeof(Params), cudaMemcpyDeviceToHost));
  EXPECT_EQ(copied.transform, transform.transform);

  CopyTransformsGPU<2, true>(d_output.get(), d_input_ptrs.get(), 1, 0);
  CUDA_CALL(cudaStreamSynchronize(0));
  Params inverted;
  CUDA_CALL(cudaMemcpy(&inverted, d_output.get(), sizeof(Params), cudaMemcpyDeviceToHost));
  EXPECT_EQ(inverted.transform, expected_inv.transform);
}

}  // namespace dali
