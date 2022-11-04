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

#include <vector>
#include <optional>
#include "dali/core/common.h"
#include "dali/kernels/imgproc/roi.h"
#include "dali/kernels/kernel.h"
#include "include/dali/core/boundary.h"
#include "include/dali/core/geom/box.h"

namespace dali {
namespace kernels {
namespace remap {

/**
 * API for Remap operation. Remap applies a generic geometrical transformation to an image.
 * @tparam Backend Storage backend for data.
 * @tparam T Type of input and output data.
 */
template <typename Backend, typename T>
struct RemapKernel {
  using Border = boundary::Boundary<T>;
  using MapType = float;
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
   * output(x,y) = input( mapx(x,y) , mapy(x,y) ).
   *
   * When BoundaryType::TRANSPARENT is used and (according to mapx and mapy) the pixels in the
   * destination image correspond to pixels outside of the source image, the destination image is
   * assigned with unmodified pixels from the source image, i.e.:
   * mapx(x, y) < 0 || mapx(x, y) > input.width => output(x, y) ::== input(x, y)
   *
   * @param context Context for the operation.
   * @param output Output batch.
   * @param input Input batch.
   * @param mapsx Arrays of floats, that determine the transformation.
   * @param mapsy Arrays of floats, that determine the transformation.
   * @param output_rois ROIs of the output images. If empty, it is assumed that the ROI covers the
   *                    whole image.
   * @param input_rois ROIs of the input images. If empty, it is assumed that the ROI covers the
   *                   whole image.
   * @param interpolations Determines, which interpolation shall be used. If empty, it is assumed
   *                       that every sample is processed with INTERP_LINEAR.
   * @param borders Determines, how to handle pixels on a border on an image (or ROI). If empty,
   *                it is assumed that every sample is processed with REFLECT_101.
   */
  virtual void Run(KernelContext &context, TensorListView<Backend, T> output,
                   TensorListView<Backend, const T> input,
                   TensorListView<Backend, const MapType, 2> mapsx,
                   TensorListView<Backend, const MapType, 2> mapsy,
                   span<const Roi<2>> output_rois = {}, span<const Roi<2>> input_rois = {},
                   span<DALIInterpType> interpolations = {}, span<Border> borders = {}) = 0;


  /**
   * Convenient overload. This function shall be used when the transformation parameters are the
   * same for every sample in an input batch.
   */
  virtual void Run(KernelContext &context, TensorListView<Backend, T> output,
                   TensorListView<Backend, const T> input,
                   TensorView<Backend, const MapType, 2> mapx,
                   TensorView<Backend, const MapType, 2> mapy,
                   Roi<2> output_roi = {}, Roi<2> input_roi = {},
                   DALIInterpType interpolation = DALI_INTERP_LINEAR, Border border = {}) = 0;
};

}  // namespace remap
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_GEOM_REMAP_H_
