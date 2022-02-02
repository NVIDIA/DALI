// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_LAPLACIAN_TEST_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_LAPLACIAN_TEST_H_

#include <array>

#include "dali/kernels/common/utils.h"
#include "dali/core/tensor_view.h"


namespace dali {
namespace kernels {

/**
 * @brief Smoothing window (d_order=0) of size 2n + 1 is [1, 2, 1] conv composed
 * with itself n - 1 times so that the window has appropriate size: it boils down
 * to computing binominal coefficients: (1 + 1) ^ (2n). Derivative kernel of the order
 * 2 is [1, -2, 1] (in other words [1, -1] composed with itself) composed with
 * the smoothing window of appropriate size
 */
void FillSobelWindow(span<float> window, int d_order);

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_LAPLACIAN_TEST_H_
