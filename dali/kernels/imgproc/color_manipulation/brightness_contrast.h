// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_BRIGHTNESS_CONTRAST_H_
#define DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_BRIGHTNESS_CONTRAST_H_

#include <utility>
#include "dali/util/ocv.h"
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/geom/box.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include "dali/pipeline/data/types.h"

namespace dali {
namespace kernels {

namespace brightness_contrast {


/**
 * Assumes HWC layout
 */
template<size_t ndims, class CoordinateType>
TensorShape<ndims + 1> roi_shape(Box<ndims, CoordinateType> roi, size_t nchannels) {
  assert(all_coords(roi.hi >= roi.lo) && "Cannot create a tensor shape from an invalid Box");
  TensorShape<ndims + 1> ret;
  auto e = roi.extent();
  auto ridx = ndims;
  ret[ridx--] = nchannels;
  for (size_t idx = 0; idx < ndims; idx++) {
    ret[ridx--] = e[idx];
  }
  return ret;
}

}  // namespace brightness_contrast


template<typename OutputType, typename InputType, size_t ndims = 3>
class BrightnessContrastCpu {
 private:
  static constexpr size_t spatial_dims = ndims - 1;

 public:
  using Roi = Box<spatial_dims, int>;


  KernelRequirements
  Setup(KernelContext &context, const InTensorCPU<InputType, ndims> &in, float brightness,
        float contrast, const Roi *roi = nullptr) {
    DALI_ENFORCE(!roi || all_coords(roi->hi >= roi->lo), "Region of interest is invalid");
    auto adjusted_roi = AdjustRoi(roi, in.shape);
    KernelRequirements req;
    TensorListShape<> out_shape({
        brightness_contrast::roi_shape(adjusted_roi, in.shape[ndims - 1])
    });
    req.output_shapes = {std::move(out_shape)};
    return req;
  }


  /**
   * Assumes HWC memory layout
   *
   * @param out Assumes, that memory is already allocated
   * @param brightness Additive brightness delta. 0 denotes no change
   * @param contrast Multiplicative contrast delta. 1 denotes no change
   * @param roi When default or invalid roi is provided,
   *            kernel operates on entire image ("no-roi" case)
   */
  void Run(KernelContext &context, const OutTensorCPU<OutputType, ndims> &out,
           const InTensorCPU<InputType, ndims> &in, float brightness, float contrast,
           const Roi *roi = nullptr) {
    auto adjusted_roi = AdjustRoi(roi, in.shape);
    auto num_channels = in.shape[2];
    auto image_width = in.shape[1];
    auto ptr = out.data;

    ptrdiff_t row_stride = image_width * num_channels;
    auto *row = in.data + adjusted_roi.lo.y * row_stride;
    for (int y = adjusted_roi.lo.y; y < adjusted_roi.hi.y; y++) {
      for (int xc = adjusted_roi.lo.x * num_channels; xc < adjusted_roi.hi.x * num_channels; xc++)
        *ptr++ = ConvertSat<OutputType>(row[xc] * contrast + brightness);
      row += row_stride;
    }
  }


 private:
  Roi AdjustRoi(const Roi *roi, const TensorShape<ndims> &shape) {
    ivec<spatial_dims> size;
    for (size_t i = 0; i < spatial_dims; i++)
      size[i] = shape[spatial_dims - 1 - i];
    Roi whole_image = {0, size};
    return roi ? intersection(*roi, whole_image) : whole_image;
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_BRIGHTNESS_CONTRAST_H_
