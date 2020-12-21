// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_PASTE_PASTE_H_
#define DALI_KERNELS_IMGPROC_PASTE_PASTE_H_

#include <utility>
#include "dali/util/ocv.h"
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/geom/box.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/imgproc/roi.h"

#define X_AXIS 0
#define Y_AXIS 1
#define C_AXIS 2

namespace dali {
namespace kernels {

template<typename OutputType, typename InputType>
class PasteCpu {
 public:
  using Roi = Box<2, int>;
  using Image = InTensorCPU<InputType, 3>;
  using Coords = InTensorCPU<const int, 1>;

  KernelRequirements
  Setup(KernelContext &context, const OutTensorCPU<OutputType> &pasteFrom, const Coords &in_anchors,
        const Coords &in_shapes, const Coords &out_anchors) {
    // KernelRequirements req;
    // TensorListShape<> out_shape({ShapeFromRoi(adjusted_roi, in.shape[ndims - 1])});
    // TensorListShape<> out_shape({h, w, inLU.shape[2]});
    // req.output_shapes = {std::move(out_shape)};
    // return req;
    return nullptr;
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
  void Run(KernelContext &context, const OutTensorCPU<OutputType> &out, const Image &pasteFrom,
           const Coords &in_anchors, const Coords &in_shapes, const Coords &out_anchors) {
    // copyRoi(out, pasteFrom, cropLU, 0, 0);
  }

  void copyRoi(const OutTensorCPU<OutputType> &out, const Image &in, int inXAnchor, int inYAnchor,
               int inXShape, int inYShape, int outXAnchor, int outYAnchor) {
    auto num_channels = out.shape[C_AXIS];
    auto in_image_width = out.shape[X_AXIS];
    auto out_image_width = in.shape[X_AXIS];

    ptrdiff_t in_row_stride = in_image_width * num_channels;
    ptrdiff_t out_row_stride = out_image_width * num_channels;

    auto *out_ptr = out.data + outXAnchor * num_channels + outYAnchor * out_row_stride;
    auto *in_ptr = in.data + (inXAnchor + inXShape) * num_channels
                   + (inYAnchor + inYShape) * in_row_stride;

    auto row_value_count = inXShape * num_channels;

    for (int y = inYAnchor; y < inYAnchor + inYShape; y++) {
      memcpy(in_ptr, out_ptr, row_value_count * sizeof(in_ptr[0]));
      // for (int xc = 0; xc < row_value_count; xc++) ptr[xc] = in_row[xc];
      in_ptr += in_row_stride;
      out_ptr += out_row_stride;
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_PASTE_PASTE_H_
