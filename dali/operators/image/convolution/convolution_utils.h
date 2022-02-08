// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  int axes;
  bool has_channels;

  inline bool operator==(const DimDesc &other) const {
    return axes == other.axes && has_channels == other.has_channels;
  }

  inline bool operator!=(const DimDesc &other) const {
    return !(*this == other);
  }
};

void ValidateLayout(int ndim, const TensorLayout &layout);
DimDesc ParseSampleLayout(int ndim, const TensorLayout &sample_layout);


}  // namespace convolution_utils
}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CONVOLUTION_CONVOLUTION_UTILS_H_
