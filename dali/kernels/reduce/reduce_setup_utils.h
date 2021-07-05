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

#ifndef DALI_KERNELS_REDUCE_REDUCE_SETUP_UTILS_H_
#define DALI_KERNELS_REDUCE_REDUCE_SETUP_UTILS_H_

/** @file
 *
 * This file contains utilities for setting up multi-stage directional reduction
 */

#include <utility>
#include <type_traits>
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/small_vector.h"
#include "dali/core/span.h"
#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/core/util.h"

namespace dali {
namespace kernels {
namespace reduce_impl {

/**
 * @brief Calculates equivalent, simpler reduction by collapsing adjacent dimensions.
 *
 * @param out_axes   a collection of integers, supporting `clear` and `push_back`.
 * @param dim_groups a collection of integer pairs, supporting `clear` and `push_back`;
 *                   the returned values are { index, count }, where index is the index of the
 *                   first dimension in group and count is the number of collapsed dimensions.
 *
 * @param in_shape   shape of the input
 * @param axes       list of reduced axes; axis indices must be <= 63
 *
 * A dimension can be collapsed with its neighbor if:
 *  - its extent is 1 in all samples
 *  - both dimensions are reduced or non-reduced.
 *
 * @remarks The function assumes that arguments are valid.
 */
template <typename Axes, typename DimGroups, int ndim>
void SimplifyReduction(Axes &out_axes, DimGroups &out_dim_groups,
                       const TensorListShape<ndim> &in_shape, span<const int> axes) {
  out_axes.clear();
  out_dim_groups.clear();
  int d = in_shape.sample_dim();

  uint64_t mask = to_bit_mask(axes);

  int i;

  // skip leading degenerate dimensions
  for (i = 0; i < d && is_degenerate_dim(in_shape, i); i++) {}

  if (i >= d) {
    // no non-degenerate dimensions
    out_dim_groups.push_back(std::make_pair(0, d));
    return;
  }

  // so we've reached the outermost non-degenerate dimension
  int remapped = 0;  // its index will map to 0

  bool prev_reduced = (mask >> i) & 1;  // is it reduced?
  if (prev_reduced)
    out_axes.push_back(remapped);  // if so, add the axis to the list of reduced dimension groups

  int group_start = 0;     // the dimension group starts at 0 -
  int group_size = i + 1;  // all leading degenerate dimensions are in this group

  for (i++; i < d; i++) {
    if (is_degenerate_dim(in_shape, i)) {
      // Degenerate dimension can be merged with the previous one regardless of whether
      // it is reduced or not.
      group_size++;
      continue;
    }
    bool is_reduced = (mask >> i) & 1;
    if (is_reduced != prev_reduced) {
      out_dim_groups.emplace_back(group_start, group_size);
      group_start = i;
      group_size = 1;
      remapped++;
      if (is_reduced)
        out_axes.push_back(remapped);
    } else {
      group_size++;
    }
    prev_reduced = is_reduced;
  }
  out_dim_groups.push_back(std::make_pair(group_start, group_size));
}


/**
 * @brief Throws an exception if given tensor list cannot be reduced across samples with
 *        given set of axes.
 *
 * A tensor list can be reduced across samples only if the non-reduced dimensions have the
 * same extent in all samples.
 */
inline void CheckBatchReduce(const TensorListShape<> &tls, span<const int> axes) {
  if (tls.num_samples() == 0)
    return;

  uint64_t mask = to_bit_mask(axes);
  SmallVector<int, DynamicTensorShapeContainer::static_size> non_reduced;
  for (int a = 0; a < tls.sample_dim(); a++) {
    if (!(mask & (1_u64 << a)))
      non_reduced.push_back(a);
  }

  auto first_sample_shape = tls.tensor_shape_span(0);

  for (int i = 1; i < tls.num_samples(); i++) {
    auto sample_shape = tls.tensor_shape_span(i);
    for (int a : non_reduced) {
      if (sample_shape[a] != first_sample_shape[a])
        throw std::logic_error(make_string(
          "Reduce: batch reduction requires that all samples have the same extent in non-reduced "
          "dimensions.\nError at sample ", i, " axis ", a, ": the extent is ", sample_shape[a],
          " != ", first_sample_shape[a], " in the first sample in the batch."));
    }
  }
}


/**
 * @brief Checks that axes only appear once and that they are within range.
 *
 * @param axes list of axis indices
 * @param ndim dimensionality of the tensor(list) to which axes refer
 */
inline void CheckAxes(span<const int> axes, int ndim) {
  assert(ndim >= 0 && ndim <= 64);
  uint64_t mask = 0;
  for (auto a : axes) {
    if (a < 0 || a >= ndim)
      throw std::out_of_range(make_string("Axis index out of range: ", a, " not in 0..", ndim-1));
    uint64_t amask = 1_u64 << a;
    if (mask & amask)
      throw std::invalid_argument(make_string("Duplicate axis index ", a));
    mask |= amask;
  }
}

/**
 * @brief Calculates the shape of the result of reduction under given parameters
 *
 * The reduced shape is calculated in one of two ways:
 * 1. If `keep_dims` is `true`, the extents of reduced dimensions in the input are replaced with 1
 * 2. If `keep_dims` is `false`, the dimensions specified in `axes` are removed from the output
 *    shape.
 *
 * If `batch_reduce` is specified, the output contains just one sample.
 *
 * @note `batch_reduce` requires that all non-reduced extents are equal in all input tensors.
 *
 * @param out_shape     shape of the output
 * @param in_shape      shape of the input to reduce
 * @param axes          indices of reduced dimensions
 * @param keep_dims     if `true`, the reduced dimensions stay in the output shape, with value 1
 *                      if `false`, the reduced dimensions are omitted in the output shape
 * @param batch_reduce  if `true`, there's just one sample in the output
 *
 * @remarks The function assumes that arguments are valid.
 */
inline void CalculateReducedShape(TensorListShape<> &out_shape,
                                  const TensorListShape<> &in_shape,
                                  span<const int> axes,
                                  bool keep_dims,
                                  bool batch_reduce) {
  int nsamples = in_shape.num_samples();
  int out_samples = batch_reduce ? 1 : nsamples;
  int in_dim = in_shape.sample_dim();
  uint64_t mask = to_bit_mask(axes);

  int out_dim = keep_dims ? in_dim : in_dim - axes.size();
  assert(out_dim >= 0);

  if (out_dim == 0) {
    out_shape.resize(out_samples, 0);
    return;
  }
  out_shape.resize(out_samples, out_dim);
  for (int i = 0; i < out_samples; i++) {
    auto in_sample_shape = in_shape.tensor_shape_span(i);
    auto out_sample_shape = out_shape.tensor_shape_span(i);
    int out_d = 0;
    for (int d = 0; d < in_dim; d++) {
      if (mask & (1ul << d)) {
        if (keep_dims)
          out_sample_shape[out_d++] = 1;
        continue;  // skip reduced axes
      }
      assert(out_d < out_dim);
      out_sample_shape[out_d++] = in_sample_shape[d];
    }
    assert(out_d == out_dim);
  }
}

/**
 * @brief Calculates reduction factor for each sample in `in_shape`
 *
 * Reduction factor is the number of input values contributing to a single reduced value.
 * Example:
 * ```
 * in_shape = { {4, 5, 6}, {1, 2, 3 } }
 * axes = { 0, 2 }
 * out = { 4*6, 1*3 }
 * ```
 * If `axes` are empty, the result is always 1. If `axes` specify all dimensions,
 * the result is an array of volumes of respective samples in in_shape.
 *
 * @param out       array of numbers, containing the reduction factors
 * @param in_shape  shape of the input
 * @param axes      indices of reduced dimensions
 *
 * @remarks The function assumes that arguments are valid.
 */
template <typename ArrayLike, int ndim>
inline void CalculateReductionFactors(ArrayLike &out, const TensorListShape<ndim> &in_shape,
                                      span<const int> axes) {
  for (int i = 0; i < in_shape.num_samples(); i++) {
    auto sample_shape = in_shape.tensor_shape_span(i);
    int64_t red = 1;
    for (auto a : axes)
      red *= sample_shape[a];
    out[i] = red;
  }
}

}  // namespace reduce_impl
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_REDUCE_REDUCE_SETUP_UTILS_H_
