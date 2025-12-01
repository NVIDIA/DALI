// Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// limitations under the License.

#include <nvimgcodec.h>
#include "dali/core/common.h"
#include "dali/core/tensor_shape.h"

#ifndef DALI_OPERATORS_IMGCODEC_IMGCODEC_H_
#define DALI_OPERATORS_IMGCODEC_IMGCODEC_H_

namespace dali {
namespace imgcodec {

struct ImageInfo {
  TensorShape<> shape;
  nvimgcodecOrientation_t orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION,
                                         sizeof(nvimgcodecOrientation_t),
                                         nullptr,
                                         0,
                                         false,
                                         false};
};

/**
 * @brief Region of interest
 *
 * Spatial coordinates of the ROI to decode. Channels shall not be included.
 *
 * If there are no coordinates for `begin` or `end` then no cropping is requried and the
 * ROI is considered to include the entire image.
 *
 * NOTE: If the orientation of the image is adjusted, these values are in the output space
 *       (after including the orientation).
 */
struct ROI {
  /**
   * @brief The beginning and end of the region-of-interest.
   *
   * If both begin and end are empty, the ROI denotes full image.
   */
  TensorShape<> begin, end;

  bool use_roi() const {
    return begin.sample_dim() || end.sample_dim();
  }

  explicit operator bool() const {
    return use_roi();
  }

  /**
   * @brief Returns the extent of the region of interest as (end - begin)
   */
  TensorShape<> shape() const {
    TensorShape<> out = end;
    assert(out.sample_dim() >= begin.sample_dim());
    for (int d = 0; d < begin.sample_dim(); d++)
      out[d] -= begin[d];
    return out;
  }
};


}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_OPERATORS_IMGCODEC_IMGCODEC_H_
