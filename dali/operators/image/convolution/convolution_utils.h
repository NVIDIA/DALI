// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_CONVOLUTION_CONVOLUTION_UTILS_H_
#define DALI_OPERATORS_IMAGE_CONVOLUTION_CONVOLUTION_UTILS_H_

#include "dali/pipeline/operator/common.h"

namespace dali {
namespace convolution_utils {

struct DimDesc {
  int usable_axes_start;
  int usable_axes_count;
  int total_axes_count;

  inline bool is_channel_last() const {
    return usable_axes_start + usable_axes_count < total_axes_count;
  }

  inline bool is_sequence() const {
    return usable_axes_start > 0;
  }

  inline bool operator==(const DimDesc &other) const {
    return usable_axes_start == other.usable_axes_start &&
           usable_axes_count == other.usable_axes_count &&
           total_axes_count == other.total_axes_count;
  }

  inline bool operator!=(const DimDesc &other) const {
    return !(*this == other);
  }
};

DimDesc ParseAndValidateDim(int ndim, const TensorLayout &layout);

}  // namespace convolution_utils
}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CONVOLUTION_CONVOLUTION_UTILS_H_
