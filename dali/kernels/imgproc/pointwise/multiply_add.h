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

#ifndef DALI_KERNELS_IMGPROC_POINTWISE_MULTIPLY_ADD_H_
#define DALI_KERNELS_IMGPROC_POINTWISE_MULTIPLY_ADD_H_

#include <utility>
#include "dali/util/ocv.h"
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/geom/box.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/imgproc/roi.h"

namespace dali {
namespace kernels {

template<typename OutputType, typename InputType, int ndims = 3>
class MultiplyAddCpu {
 private:
  static constexpr size_t spatial_dims = ndims - 1;

 public:
  using Roi = Box<spatial_dims, int>;


  KernelRequirements
  Setup(KernelContext &context, const InTensorCPU<InputType, ndims> &in, float addend,
        float multiplier, const Roi *roi = nullptr) {
    DALI_ENFORCE(!roi || all_coords(roi->hi >= roi->lo), "Region of interest is invalid");
    auto adjusted_roi = AdjustRoi(roi, in.shape);
    KernelRequirements req;
    TensorListShape<> out_shape({ShapeFromRoi(adjusted_roi, in.shape[ndims - 1])
    });
    req.output_shapes = {std::move(out_shape)};
    return req;
  }


  /**
   * Assumes HWC memory layout
   *
   * @param out Assumes, that memory is already allocated
   * @param addend Additive addend delta. 0 denotes no change
   * @param multiplier Multiplicative multiplier delta. 1 denotes no change
   * @param roi When default or invalid roi is provided,
   *            kernel operates on entire image ("no-roi" case)
   */
  void Run(KernelContext &context, const OutTensorCPU<OutputType, ndims> &out,
           const InTensorCPU<InputType, ndims> &in, float addend, float multiplier,
           const Roi *roi = nullptr) {
    auto adjusted_roi = AdjustRoi(roi, in.shape);
    auto num_channels = in.shape[2];
    auto image_width = in.shape[1];
    auto ptr = out.data;

    ptrdiff_t row_stride = image_width * num_channels;
    auto *row = in.data + adjusted_roi.lo.y * row_stride;
    for (int y = adjusted_roi.lo.y; y < adjusted_roi.hi.y; y++) {
      for (int xc = adjusted_roi.lo.x * num_channels; xc < adjusted_roi.hi.x * num_channels; xc++)
        *ptr++ = ConvertSat<OutputType>(row[xc] * multiplier + addend);
      row += row_stride;
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_POINTWISE_MULTIPLY_ADD_H_
