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

#ifndef DALI_BRIGHTNESS_CONTRAST_CPU_H
#define DALI_BRIGHTNESS_CONTRAST_CPU_H

#include "dali/util/ocv.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/imgproc/imgproc_common.h"
#include "dali/pipeline/data/types.h"

namespace dali {
namespace kernels {

template<typename ComputeBackend, typename InputType, typename OutputType>
class DLL_PUBLIC BrightnessContrast {
 private:
  using StorageBackend = compute_to_storage_t<ComputeBackend>;
 public:

  DLL_PUBLIC KernelRequirements
  Setup(KernelContext &context, const InTensor<StorageBackend, InputType, 3> &image,InputType brightness,
        InputType contrast, Roi roi = {0, 0, 0, 0}) {
    //TODO validate roi
    handle_default_roi(roi, image.shape);
    KernelRequirements req;
    req.output_shapes = {TensorListShape<DynamicDimensions>({roi_to_shape<3>(roi)})};
    return req;
  }


  //TODO CHW layout
  //TODO first brightness then contrast
  /**
   * Assumes (for now) HWC memory layout
   *
   * @param out Assumes, that memory is already allocated
   * @param brightness Additive brightness delta. 0 denotes no change
   * @param contrast Multiplicative contrast delta. 1 denotes no change
   * @param roi When default roi is provided, kernel operates on entire image ("no-roi" case)
   */
  DLL_PUBLIC void Run(KernelContext &context, const OutTensor<StorageBackend, OutputType, 3> &out,const InTensor<StorageBackend, InputType, 3> &in,
                       InputType brightness,
                      InputType contrast, Roi roi = {0, 0, 0, 0}) {
    handle_default_roi(roi, in.shape);
    size_t num_channels = in.shape[2];
    size_t W = in.shape[1];
    auto ptr = out.data;
    DALI_ENFORCE(roi.h > 0 && roi.w > 0, "Region of interest can't be empty");

//    for (size_t y = 0; y < roi.h; y++) {
//      for (size_t x = 0; x < roi.w; x++) {
//        for (size_t c = 0; c < num_channels; c++) {
//          out.data[roi.w * y * num_channels + x * num_channels + c] =
//                  in.data[roi.w * y * num_channels + x * num_channels + c] * contrast + brightness;
//        }
//      }
//    }

    for (int y=roi.y;y<roi.y+roi.h;y++) {
      for (int xc=(roi.x+y*W)*3;xc<(roi.x+roi.w+y*W)*3;xc++) {
        *ptr++=in.data[xc]*contrast+brightness;
      }
    }

  }


 private:
  void handle_default_roi(Roi &roi, const TensorShape<DynamicDimensions> &shape) {
    if (roi.h == 0 && roi.w == 0) {
      roi.x = 0;
      roi.y = 0;
      roi.h = shape[0];
      roi.w = shape[1];
    }
  }

  template<int nchannels>
  TensorShape<nchannels> roi_to_shape(const Roi &roi) {
    TensorShape<nchannels> sh = {static_cast<int64_t>(roi.h), roi.w, nchannels};
    return sh;
  }
};

}  // namespace kernels
}  // namespace dali

#endif //DALI_BRIGHTNESS_CONTRAST_CPU_H
