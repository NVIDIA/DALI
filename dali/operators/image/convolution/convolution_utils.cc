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

#include <functional>

#include "dali/operators/image/convolution/convolution_utils.h"
#include "dali/pipeline/operator/common.h"

namespace dali {
namespace convolution_utils {

DimDesc ParseAndValidateDim(int ndim, const TensorLayout &layout) {
  static constexpr int kMaxDim = 3;
  if (layout.empty()) {
    // assuming plain data with no channels
    DALI_ENFORCE(ndim <= kMaxDim,
                 make_string("Input data with empty layout cannot have more than ", kMaxDim,
                             " dimensions, got input with ", ndim, " dimensions."));
    return {0, ndim, ndim};
  }
  int axes_start = 0;
  int axes_end = ndim;
  while (axes_start < ndim && (layout[axes_start] == 'C' || layout[axes_start] == 'F')) {
    axes_start++;
  }
  if (axes_end > 0 && layout[axes_end - 1] == 'C') {
    axes_end--;
  }
  int axes_count = axes_end - axes_start;
  DALI_ENFORCE(axes_count > 0, make_string("No spatial axes found in the layout: ", layout));
  DALI_ENFORCE(
      std::all_of(layout.begin() + axes_start, layout.begin() + axes_end,
                  std::bind(std::not_equal_to<char>(), 'C', std::placeholders::_1)),
      make_string("Only channel-first or channel-last layouts are supported, got: ", layout, "."));
  DALI_ENFORCE(
      std::all_of(layout.begin() + axes_start, layout.begin() + axes_end,
                  std::bind(std::not_equal_to<char>(), 'F', std::placeholders::_1)),
      make_string("For sequences, layout should begin with 'F' or 'C', got: ", layout, "."));
  DALI_ENFORCE(
      axes_start <= 2,
      make_string("Found more the one occurrence of 'F' or 'C' axes in layout: ", layout, "."));
  DALI_ENFORCE(axes_count <= kMaxDim,
               make_string("Too many dimensions, found: ", axes_count,
                           " data axes, maximum supported is: ", kMaxDim, "."));
  return {axes_start, axes_count, ndim};
}

}  // namespace convolution_utils
}  // namespace dali
