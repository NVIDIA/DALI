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

}  // namespace warp_perspective
}  // namespace dali
