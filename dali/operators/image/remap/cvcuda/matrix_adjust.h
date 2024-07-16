// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_REMAP_CVCUDA_MATRIX_ADJUST_H_
#define DALI_OPERATORS_IMAGE_REMAP_CVCUDA_MATRIX_ADJUST_H_

#include <dali/core/geom/mat.h>
#include <dali/pipeline/data/tensor.h>
#include <nvcv/Tensor.hpp>

namespace dali {
namespace warp_perspective {

/**
 * @brief Modifies (in-place) tensor of perspective matrices to match
 * the OpenCV convention of pixel origin (center instead of corner).
 */
void adjustMatrices(nvcv::Tensor &matrices, cudaStream_t stream);

}  // namespace warp_perspective
}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_REMAP_CVCUDA_MATRIX_ADJUST_H_
