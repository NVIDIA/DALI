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

#ifndef DALI_KERNELS_IMGPROC_COLOUR_MANIPULATION_BRIGHTNESS_CONTRAST_H_
#define DALI_KERNELS_IMGPROC_COLOUR_MANIPULATION_BRIGHTNESS_CONTRAST_H_

#include "dali/util/ocv.h"
#include "dali/core/common.h"
#include "dali/core/geom/box.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include "dali/pipeline/data/types.h"

namespace dali {
namespace kernels {

namespace brightness_contrast {

// TODO(mszolucha): Add Box support -> Roi(const Box<3, int> &box)
struct Roi {
  Roi() = default;


  Roi(int x, int y, int w, int h) : x(x), y(y), w(w), h(h) {
    if (w > 0 && h > 0) {
      valid_roi = true;
    }
  }


  int x, y, w, h;
  bool valid_roi = false;
};

}  // namespace brightness_contrast

template<typename ComputeBackend, typename InputType, typename OutputType>
class DLL_PUBLIC BrightnessContrast {
 private:
  using StorageBackend = compute_to_storage_t<ComputeBackend>;
  static constexpr size_t ndims = 3;

 public:
  DLL_PUBLIC KernelRequirements
  Setup(KernelContext &context, const InTensor<StorageBackend, InputType, ndims> &image,
        InputType brightness, InputType contrast, brightness_contrast::Roi roi = {}) {
    handle_invalid_roi(roi, image.shape);
    KernelRequirements req;
    req.output_shapes = {TensorListShape<DynamicDimensions>({roi_to_shape<ndims>(roi)})};
    return req;
  }


  // TODO(mszolucha): CHW layout
  // TODO(mszolucha): first brightness then contrast
  /**
   * Assumes HWC memory layout
   *
   * @param out Assumes, that memory is already allocated
   * @param brightness Additive brightness delta. 0 denotes no change
   * @param contrast Multiplicative contrast delta. 1 denotes no change
   * @param roi When default or invalid roi is provided,
   *            kernel operates on entire image ("no-roi" case)
   */
  DLL_PUBLIC void
  Run(KernelContext &context, const OutTensor<StorageBackend, OutputType, ndims> &out,
      const InTensor<StorageBackend, InputType, ndims> &in, InputType brightness,
      InputType contrast, brightness_contrast::Roi roi = {}) {
    handle_invalid_roi(roi, in.shape);
    auto num_channels = in.shape[2];
    auto image_width = in.shape[1];
    auto ptr = out.data;

    for (int y = roi.y; y < roi.y + roi.h; y++) {
      for (int xc = (roi.x + y * image_width) * num_channels;
           xc < (roi.x + roi.w + y * image_width) * num_channels; xc++) {
        *ptr++ = in.data[xc] * contrast + brightness;
      }
    }
  }


 private:
  void
  handle_invalid_roi(brightness_contrast::Roi &roi, const TensorShape<DynamicDimensions> &shape) {
    if (!roi.valid_roi) {
      roi.x = 0;
      roi.y = 0;
      roi.h = shape[0];
      roi.w = shape[1];
      roi.valid_roi = true;
    }
  }


  template<int nchannels>
  TensorShape<nchannels> roi_to_shape(const brightness_contrast::Roi &roi) {
    TensorShape<nchannels> sh = {static_cast<int64_t>(roi.h), roi.w, nchannels};
    return sh;
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOUR_MANIPULATION_BRIGHTNESS_CONTRAST_H_
