// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_MATH_EXPRESSIONS_BROADCASTING_H_
#define DALI_OPERATORS_MATH_EXPRESSIONS_BROADCASTING_H_

#include <utility>
#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_shape_print.h"

namespace dali {

/**
 * @brief Calculates the shape of broadcasting two compatible shapes. It is allowed that the shape
 *        has different number of dimensions. In such case, the shape with fewer dimensions will be
 *        prepended with leading dimensions with extend 1.
 *        Example: lhs=(10, 10, 3), rhs=(1, 3) -> result=(10, 10, 3)
 * 
 * @param lhs shape of left hand side operand
 * @param rhs shape of right hand side operand
 * @return TensorShape<> resulting shape
 */
TensorShape<> BroadcastShape(const TensorShape<> &lhs, const TensorShape<> &rhs) {
  TensorShape<> sh;

  const int64_t *a = lhs.shape.data(), *b = rhs.shape.data();
  size_t a_ndim = lhs.shape.size(), b_ndim = rhs.shape.size();
  if (a_ndim < b_ndim) {
    std::swap(a, b);
    std::swap(a_ndim, b_ndim);
  }

  size_t i = 0, j = 0;
  for (; i < (a_ndim - b_ndim); i++) {
    sh.shape.push_back(a[i]);
  }
  for (; i < a_ndim; i++, j++) {
    DALI_ENFORCE(a[i] == b[j] || a[i] == 1 || b[j] == 1,
                 make_string("Shapes ", lhs, " and ", rhs, " can't be broadcasted (",
                             a[i], ", ", b[j], ")."));
    sh.shape.push_back(std::max(a[i], b[j]));
  }
  return sh;
}

/**
 * @brief Calculates strides to cover a possibly broadcasted shape. Those stride for
 *        broadcasted dimensions is changed to 0
 * 
 * @param out_sh broadcasted shape
 * @param in_sh original shape
 * @param in_strides original strides
 * @return TensorShape<> modified strides
 */
TensorShape<> StridesForBroadcasting(const TensorShape<> &out_sh, const TensorShape<> &in_sh,
                                     const TensorShape<> &in_strides) {
  TensorShape<> strides;
  assert(in_sh.size() == in_strides.size());
  assert(in_sh.size() <= out_sh.size());
  int i = 0;
  int out_ndim = out_sh.size();
  strides.shape.resize(out_ndim, 0);
  for (int i = (out_ndim - in_sh.size()); i < out_ndim; i++) {
    assert(in_sh[i] == out_sh[i] || in_sh[i] == 1);
    if (in_sh[i] == out_sh[i])
      strides[i] = in_strides[i];
    else
      strides[i] = 0;
  }
  return strides;
}

void ExpandToNDims(TensorShape<> &sh, int ndim) {
  assert(sh.size() <= ndim);
  if (sh.size() == ndim)
    return;
  TensorShape<> sh2;
  sh2.shape.resize(ndim);
  int i = 0;
  for (; i < (ndim - sh.size()); i++)
    sh2[i] = 1;
  for (int j = 0; j < sh.size(); j++, i++)
    sh2[i] = sh[j];
  std::swap(sh, sh2);
}

/**
 * @brief It simplifies a shape for arithmetic op execution with broadcasting.
 *        It detects and collapses adjacent dimensions that are not broadcasted
 * 
 */
void SimplifyShapesForBroadcasting(TensorShape<>& lhs, TensorShape<> &rhs) {
  // First, if needed expand dimensions
  int full_ndim = std::max(lhs.size(), rhs.size());
  if (lhs.size() != rhs.size()) {
    ExpandToNDims(lhs, full_ndim);
    ExpandToNDims(rhs, full_ndim);
  }

  int i = 0;
  SmallVector<std::pair<int, int>, 5> group_dims;
  while (i < full_ndim) {
    if (lhs[i] != rhs[i]) {
      i++;
      continue;
    }
    int j = i;
    for (; j < full_ndim; j++) {
      if (lhs[j] != rhs[j]) break;
    }
    if (i < j) {
      group_dims.emplace_back(i, j-i);
    }
    i = j;
  }

  lhs = collapse_dims(lhs, group_dims);
  rhs = collapse_dims(rhs, group_dims);
}

}  // namespace dali

#endif  // DALI_OPERATORS_MATH_EXPRESSIONS_BROADCASTING_H_
