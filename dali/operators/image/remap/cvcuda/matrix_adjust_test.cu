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
#include "dali/operators/image/remap/cvcuda/matrix_adjust.h"
#include "dali/operators/nvcvop/nvcvop.h"
#include "dali/core/cuda_error.h"

namespace dali {
namespace warp_perspective {

// Regression test for https://github.com/NVIDIA/DALI/issues/6504
// An empty (zero-batch) matrix tensor used to cause a host-side division by zero while
// computing the CUDA launch grid, before any kernel was launched.
TEST(MatrixAdjustTest, AdjustMatricesDoesNotCrashOnZeroBatch) {
  nvcv::Tensor matrices(nvcv::TensorShape({0, 9}, "NW"), nvcvop::GetDataType<float>());
  ASSERT_NO_THROW(adjustMatrices(matrices, 0));
  CUDA_CALL(cudaStreamSynchronize(0));
}

// Companion test that exercises adjustMatrices with a non-empty batch and checks the actual
// output against a hand-computed reference, so the zero-batch guard above can't be satisfied
// by a kernel that silently does nothing for every input.
//
// adjustMatrices re-centers the matrix on pixel centers by conjugating with a +0.5/+0.5
// shift: M' = Shift(-0.5) * M * Shift(+0.5). For a 2x uniform scale about the origin, that
// shift is not absorbed for free (unlike a pure translation), so the expected output has a
// non-trivial translation component that a broken/no-op kernel would not reproduce.
TEST(MatrixAdjustTest, AdjustMatricesAppliesPixelCenterShiftToSingleSample) {
  // clang-format off
  const float input[9] = {2, 0, 0,
                           0, 2, 0,
                           0, 0, 1};
  const float expected[9] = {2, 0, 0.5f,
                              0, 2, 0.5f,
                              0, 0, 1};
  // clang-format on

  nvcv::Tensor matrices(nvcv::TensorShape({1, 9}, "NW"), nvcvop::GetDataType<float>());
  auto data = *matrices.exportData<nvcv::TensorDataStridedCuda>();
  auto *d_ptr = reinterpret_cast<float *>(data.basePtr());
  CUDA_CALL(cudaMemcpy(d_ptr, input, sizeof(input), cudaMemcpyHostToDevice));

  adjustMatrices(matrices, 0);
  CUDA_CALL(cudaStreamSynchronize(0));

  float result[9];
  CUDA_CALL(cudaMemcpy(result, d_ptr, sizeof(result), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 9; i++) {
    EXPECT_FLOAT_EQ(result[i], expected[i]) << "at index " << i;
  }
}

}  // namespace warp_perspective
}  // namespace dali
