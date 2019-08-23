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

#ifndef DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_HSV_CPU_H_
#define DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_HSV_CPU_H_

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

namespace hsv {

constexpr size_t kNdims = 3;
constexpr size_t kNchannels = 3;

/**
 * Defines region of interest.
 * 0 dimension is interpreted along x axis (horizontal)
 * 1 dimension is interpreted along y axis (vertical)
 *
 *            image.x ->
 *          +--------------------------------+
 *          |        roi.x                   |
 *  image.y |      +-----+                   |
 *       |  | roi.y|     |                   |
 *       v  |      +-----+                   |
 *          +--------------------------------+
 */
template <size_t ndims>
using Roi = Box<ndims, int>;


/**
 * Defines TensorShape corresponding to provided Roi.
 * Assumes HWC memory layout
 *
 * @tparam ndims_roi Number of dims in Roi
 * @param roi Region of interest
 * @param nchannels Number of channels in data
 * @return Corresponding TensorShape
 */
template <size_t ndims_roi>
TensorShape<ndims_roi + 1> roi_shape(Roi<ndims_roi> roi, size_t nchannels) {
  assert(all_coords(roi.hi >= roi.lo) && "Cannot create a tensor shape from an invalid Box");
  TensorShape<ndims_roi + 1> ret;
  auto e = roi.extent();
  auto ridx = ndims_roi;
  ret[ridx--] = nchannels;
  for (size_t idx = 0; idx < ndims_roi; idx++) {
    ret[ridx--] = e[idx];
  }
  return ret;
}

}  // namespace hsv


template <class OutputType, class InputType>
class HsvCpu {
  // TODO(mszolucha): implement float16
  static_assert(!std::is_same<OutputType, float16_cpu>::value &&
                !std::is_same<InputType, float16_cpu>::value, "float16 not implemented yet");

 public:
  using Roi = hsv::Roi<hsv::kNdims - 1>;


  KernelRequirements
  Setup(KernelContext &context, const InTensorCPU<InputType, hsv::kNdims> &in, float hue,
        float saturation, float value, const Roi *roi = nullptr) {
    DALI_ENFORCE(!roi || all_coords(roi->hi >= roi->lo), "Region of interest is invalid");
    auto adjusted_roi = AdjustRoi(roi, in.shape);
    KernelRequirements req;
    TensorListShape<> out_shape({hsv::roi_shape(adjusted_roi, hsv::kNchannels)});
    req.output_shapes = {std::move(out_shape)};
    return req;
  }


  void Run(KernelContext &context, const OutTensorCPU<OutputType, hsv::kNdims> &out,
           const InTensorCPU<InputType, hsv::kNdims> &in, float hue, float saturation, float value,
           const Roi *roi = nullptr) {
    auto adjusted_roi = AdjustRoi(roi, in.shape);
    auto num_channels = in.shape[2];
    auto image_width = in.shape[1];
    auto ptr = out.data;

    ptrdiff_t row_stride = image_width * num_channels;
    auto *row = in.data + adjusted_roi.lo.y * row_stride;
    for (int y = adjusted_roi.lo.y; y < adjusted_roi.hi.y; y++) {
      for (int x = adjusted_roi.lo.x; x < adjusted_roi.hi.x; x++) {
        auto elem = row + x * num_channels;
        *ptr++ = ConvertSat<OutputType>(*(elem + 0) + hue /*hue hue*/);
        *ptr++ = ConvertSat<OutputType>(*(elem + 1) * saturation);
        *ptr++ = ConvertSat<OutputType>(*(elem + 2) * value);
      }
      row += row_stride;
    }
  }


 private:
  /**
   * Adjusted Roi is a Roi, which doesn't overflow the image,
   * that given by TensorShape.
   *
   * In case, when no Roi is provided (roi == nullptr),
   * size of whole image is returned as Roi.
   *
   * Assumes HWC layout
   */
  Roi AdjustRoi(const Roi *roi, const TensorShape<hsv::kNdims> &shape) {
    constexpr size_t spatial_dims = hsv::kNdims - 1;
    ivec<spatial_dims> size;
    for (size_t i = 0; i < spatial_dims; i++)
      size[i] = shape[spatial_dims - 1 - i];
    Roi whole_image = {0, size};
    return roi ? intersection(*roi, whole_image) : whole_image;
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_HSV_CPU_H_
