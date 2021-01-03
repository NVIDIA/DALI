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
  using Image = InTensorCPU<InputType, 3>;
  using OutImage = OutTensorCPU<OutputType>;
  // TODO(TheTimemaster): Change MultiPaste to support InTensorCPU<const int, 1> as Coords;
  using Coords = const int*;

  KernelRequirements
  Setup(KernelContext &context, const OutTensorCPU<OutputType> &pasteFrom, const Coords &in_anchors,
        const Coords &in_shapes, const Coords &out_anchors) {
    // Kernel is not aware of the size of the output. This should be handled in the operator.
    return nullptr;
  }


  /**
   * Assumes HWC memory layout
   *
   * @param out         Assumes, that memory is already allocated
   * @param pasteFrom   Input image data.
   * @param in_anchors  Pointer to two integers denoting where the selected part starts.
   * @param in_shapes   Pointer to two integers denoting what size is the selected part.
   * @param out_anchors Pointer to two integers denoting where to paste selected part.
   */
  void Run(KernelContext &context, const OutImage &out, const Image &pasteFrom,
           const Coords &in_anchors, const Coords &in_shapes, const Coords &out_anchors) {
    copyRoi(out, pasteFrom, in_anchors[X_AXIS], in_anchors[Y_AXIS],
            in_shapes[X_AXIS], in_shapes[Y_AXIS],
            out_anchors[X_AXIS], out_anchors[Y_AXIS]);
  }

  void copyRoi(const OutTensorCPU<OutputType> &out, const Image &in, int inXAnchor, int inYAnchor,
               int inXShape, int inYShape, int outXAnchor, int outYAnchor) {
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
