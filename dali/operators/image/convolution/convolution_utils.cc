// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/operator/common.h"
#include "dali/operators/image/convolution/convolution_utils.h"

namespace dali {
namespace convolution_utils {

DimDesc ParseAndValidateDim(int ndim, TensorLayout layout) {
  static constexpr int kMaxDim = 3;
  if (layout.empty()) {
    // assuming plain data with no channels
    DALI_ENFORCE(ndim <= kMaxDim,
                 make_string("Input data with empty layout cannot have more than ", kMaxDim,
                             " dimensions, got input with ", ndim, " dimensions."));
    return {0, ndim, ndim, false, false};
  }
  // not-empty layout
  int axes_start = 0;
  int axes_count = ndim;
  bool has_channels = ImageLayoutInfo::IsChannelLast(layout);
  if (has_channels) {
    axes_count--;
  }
  // Skip possible occurrences of 'C' or 'F' at the beggining
  TensorLayout layout_tmp = layout;
  while (ImageLayoutInfo::IsChannelFirst(layout_tmp) || VideoLayoutInfo::IsSequence(layout_tmp)) {
    axes_start++;
    axes_count--;
    layout_tmp = layout_tmp.sub(1);
  }
  if (!has_channels) {
    DALI_ENFORCE(!ImageLayoutInfo::HasChannel(layout_tmp),
                 make_string("Only channel-first or channel-last layouts are supported, got: ",
                             layout, "."));
  }
  DALI_ENFORCE(
      !VideoLayoutInfo::HasSequence(layout_tmp),
      make_string("For sequences, layout should begin with 'F' or 'CF', got: ", layout, "."));
  DALI_ENFORCE(
      axes_start <= 2,
      make_string("Found more the one occurrence of 'F' or 'C' axes in layout: ", layout, "."));
  DALI_ENFORCE(axes_count <= kMaxDim,
               make_string("Too many dimensions, found: ", axes_count,
                           " data axes, maximum supported is: ", kMaxDim, "."));
  return {axes_start, axes_count, axes_count + (axes_start != 0), has_channels, axes_start != 0};
}

}  // namespace convolution_utils
}  // namespace dali
