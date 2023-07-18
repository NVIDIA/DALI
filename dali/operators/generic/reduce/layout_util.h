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

namespace dali {
namespace reduce_util {

/**
 * @brief Removes the specified axes from the layout
 *
 * @param layout The layout to remove axes from. Max supported layout size is 64.
 * @param axes list of unique axes from [0...layout.size() - 1] range describing
 *             what axes to remove from the layout
 */
template <typename Axes>
inline TensorLayout ReduceLayout(const TensorLayout &layout, Axes &&axes) {
  int in_ndim = layout.size();
  assert(in_ndim >= 0 && in_ndim <= 64);
  assert(axes.size() <= in_ndim);
  uint64_t mask = 0;
  for (auto a : axes) {
    assert(0 <= a && a < in_ndim);
    uint64_t a_mask = 1_u64 << a;
    assert(!(mask & a_mask));  // axes must be unique for the correct out layout dim
    mask |= a_mask;
  }
  int out_ndim = in_ndim - axes.size();
  TensorLayout out_layout;
  if (out_ndim <= 0) {
    return out_layout;
  }
  out_layout.resize(out_ndim);
  for (int in_idx = 0, out_idx = 0; in_idx < in_ndim; in_idx++) {
    if (!(mask & (1_u64 << in_idx))) {
      out_layout[out_idx++] = layout[in_idx];
    }
  }
  return out_layout;
}

template <typename Input, typename Output, typename Axes>
inline void PropagateLayout(const Input &input, Output &output, Axes &&axes, bool keep_dims) {
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
