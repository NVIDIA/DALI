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

#ifndef DALI_KERNELS_IMGPROC_GEOM_REMAP_H_
#define DALI_KERNELS_IMGPROC_GEOM_REMAP_H_

#include <optional>
#include "dali/core/common.h"
#include "dali/kernels/kernel.h"
#include "include/dali/core/geom/box.h"

namespace dali {
namespace kernels {

/**
 * Specifies, how to handle pixels on a border of an image.
 *
 * --------------------------------------------------------------|
 * | CONSTANT    | iiiiii|abcdefgh|iiiiiii with some specified i |
 * --------------------------------------------------------------|
 * | REPLICATE   | aaaaaa|abcdefgh|hhhhhhh                       |
 * --------------------------------------------------------------|
 * | REFLECT     | fedcba|abcdefgh|hgfedcb                       |
 * --------------------------------------------------------------|
 * | REFLECT_101 | gfedcb|abcdefgh|gfedcba                       |
 * --------------------------------------------------------------|
 * | WRAP        | cdefgh|abcdefgh|abcdefg                       |
 * --------------------------------------------------------------|
 * | TRANSPARENT | uvwxyz|abcdefgh|ijklmno                       |
 * --------------------------------------------------------------|
 * | ISOLATED    | do not look outside of ROI                    |
 * --------------------------------------------------------------|
 */
enum class BorderType {
  CONSTANT,
  REPLICATE,
  REFLECT,
  REFLECT_101,
  WRAP,
  TRANSPARENT,
  ISOLATED
};

template <typename T>
struct Border {
  BorderType type;
  std::optional<T> value;
};

/**
 * API for Remap operation. Remap applies a generic geometrical transformation to an image.
 * @tparam Backend
 * @tparam T
 */
template <typename Backend, typename T>
struct RemapKernel {
  /**
   * Perform remap algorithm. This function allows to apply a different transformation to every
   * sample in a batch. For a special case, where the same transformation is applied for every
   * sample in a batch, please refer to corresponding overload.
   *
   * Every output sample has to have the same shape as the corresponding input sample.
   *
   * Handles only HWC layout.
   *
   * The transformation is described by `mapx` and `mapy` parameters, where:
   * output(x,y) = input( mapx(x,y) , mapy(x,y) )
   *
   * When a ROI is empty, it is assumed that it covers whole input image.
   *
   * @param context Context for the operation.
   * @param output Output batch.
   * @param input Input batch.
   * @param output_rois ROIs of the output images. Shall be the same size as input_rois.
   * @param input_rois ROIs of the input images.
   * @param mapsx Arrays of floats, that determine the transformation.
   * @param mapsy Arrays of floats, that determine the transformation.
   * @param interpolations Determines, which interpolation shall be used.
   * @param borders Determines, how to handle pixels on a border on an image (or ROI).
   */
  void Run(KernelContext &context, TensorListView<Backend, T> output,
           TensorListView<Backend, const T> input, span<Box<2, int64_t>> output_rois,
           span<Box<2, int64_t>> input_rois, span<TensorView<Backend, const float, 2>> mapsx,
           span<TensorView<Backend, const float, 2>> mapsy, span<DALIInterpType> interpolations,
           span<Border<T>> borders) = 0;


  /**
   * Convenient overload. This function shall be used when the transformation parameters are the
   * same for every sample in an input batch.
   */
  void Run(KernelContext &context, TensorListView<Backend, T> output, Box<2, int64_t> output_roi,
           TensorListView<Backend, const T> input, Box<2, int64_t> input_roi,
           TensorView<Backend, const float, 2> mapx, TensorView<Backend, const float, 2> mapy,
           DALIInterpType interpolation, Border<T> border) = 0;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_GEOM_REMAP_H_
