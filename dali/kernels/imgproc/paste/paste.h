// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

const int Y_AXIS = 0;
const int X_AXIS = 1;
const int C_AXIS = 2;

namespace dali {
namespace kernels {

template<typename OutputType, typename InputType>
class PasteCPU {
 public:
  using Image = InTensorCPU<InputType, 3>;
  using OutImage = OutTensorCPU<OutputType, 3>;
  using Coords = InTensorCPU<const int, 1>;

  /**
   * @brief Pastes regions of inputs onto the output.
   *
   * @param out         Output image data
   * @param pasteFrom   Input image data.
   * @param in_anchors  1D int view denoting where the selected part starts.
   * @param in_shapes   1D int view denoting what size is the selected part.
   * @param out_anchors 1D int view denoting where to paste selected part.
   */
  void Run(KernelContext &context, const OutImage &out, const Image &pasteFrom,
           const Coords &in_anchors, const Coords &in_shapes, const Coords &out_anchors) {
    CopyRoi(out, out_anchors.data[X_AXIS], out_anchors.data[Y_AXIS],
            pasteFrom, in_anchors.data[X_AXIS], in_anchors.data[Y_AXIS],
            in_shapes.data[X_AXIS], in_shapes.data[Y_AXIS]);
  }

  void CopyRoi(const OutImage &out, int outXAnchor, int outYAnchor, const Image &in,
               int inXAnchor, int inYAnchor, int inXShape, int inYShape) {
    auto num_channels = out.shape[C_AXIS];
    auto in_image_width = in.shape[X_AXIS];
    auto out_image_width = out.shape[X_AXIS];

    ptrdiff_t in_row_stride = in_image_width * num_channels;
    ptrdiff_t out_row_stride = out_image_width * num_channels;

    auto *out_ptr = out.data + outXAnchor * num_channels + outYAnchor * out_row_stride;
    auto *in_ptr = in.data + inXAnchor * num_channels + inYAnchor * in_row_stride;

    auto row_value_count = inXShape * num_channels;

    for (int y = 0; y < inYShape; y++) {
      if (std::is_same<InputType, OutputType>::value) {
        memcpy(out_ptr, in_ptr, row_value_count * sizeof(in_ptr[0]));
      } else {
        for (int i = 0; i < row_value_count; i++) {
          out_ptr[i] = ConvertSat<OutputType>(in_ptr[i]);
        }
      }
      in_ptr += in_row_stride;
      out_ptr += out_row_stride;
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_PASTE_PASTE_H_
