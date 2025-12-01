// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_GENERIC_REDUCE_LAYOUT_UTIL_H__
#define DALI_OPERATORS_GENERIC_REDUCE_LAYOUT_UTIL_H__

#include "dali/core/tensor_layout.h"
#include "dali/core/util.h"

namespace dali {
namespace reduce_util {

/**
 * @brief Removes the specified axes from the layout
 *
 * @param layout The layout to remove axes from. Max supported layout size is 64.
 * @param axes list of axes from [0...layout.size() - 1] range describing
 *             what axes to remove from the layout
 */
template <typename Axes>
inline TensorLayout ReduceLayout(const TensorLayout &layout, Axes &&axes) {
  uint64_t mask = to_bit_mask(axes);
  TensorLayout out_layout;
  for (int idx = 0; idx < layout.size(); idx++) {
    if (!(mask & (1_u64 << idx))) {
      out_layout += layout[idx];
    }
  }
  return out_layout;
}

template <typename Output, typename Input, typename Axes>
inline void PropagateLayout(Output &output, const Input &input, Axes &&axes, bool keep_dims) {
  const auto &in_layout = input.GetLayout();
  if (!in_layout.empty() && output.GetLayout().empty()) {
    if (keep_dims) {
      output.SetLayout(in_layout);
    } else {
      output.SetLayout(ReduceLayout(in_layout, axes));
    }
  }
}

}  // namespace reduce_util
}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_REDUCE_LAYOUT_UTIL_H__
