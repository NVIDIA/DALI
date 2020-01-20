// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_COMMON_TRANSPOSE_H_
#define DALI_KERNELS_COMMON_TRANSPOSE_H_

#include <type_traits>
#include <utility>
#include <vector>

#include "dali/core/static_switch.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/common/utils.h"

namespace dali {
namespace kernels {

namespace detail {

template <typename T, int LevelsLeft, int MaxLevels = -1>
std::enable_if_t<LevelsLeft == 1> transpose_impl_static(T *dst, const T *src, int max_levels,
                                                        span<const int64_t> dst_stride,
                                                        span<const int64_t> src_stride,
                                                        TensorShape<> size, span<const int> perm) {
  const auto level = MaxLevels >= 0 ? (MaxLevels - LevelsLeft) : (max_levels - LevelsLeft);
  auto dst_level_stride = dst_stride[level];
  auto src_level_stride = src_stride[perm[level]];
  for (int64_t i = 0; i < size[level]; i++) {
    *dst = *src;
    dst += dst_level_stride;
    src += src_level_stride;
  }
}

template <typename T, int LevelsLeft, int MaxLevels = -1>
std::enable_if_t<(LevelsLeft > 1)> transpose_impl_static(T *dst, const T *src, int max_levels,
                                                         span<const int64_t> dst_stride,
                                                         span<const int64_t> src_stride,
                                                         TensorShape<> size, span<const int> perm) {
  const auto level = MaxLevels >= 0 ? (MaxLevels - LevelsLeft) : max_levels - LevelsLeft;
  auto dst_level_stride = dst_stride[level];
  auto src_level_stride = src_stride[perm[level]];
  for (int64_t i = 0; i < size[level]; i++) {
    transpose_impl_static<T, LevelsLeft - 1, MaxLevels>(dst, src, max_levels, dst_stride,
                                                        src_stride, size, perm);
    dst += dst_level_stride;
    src += src_level_stride;
  }
}

template <typename T, int Static = 3>
void transpose_impl(T *dst, const T *src, int level, int max_levels, span<const int64_t> dst_stride,
                    span<const int64_t> src_stride, TensorShape<> size, span<const int> perm) {
  auto dst_level_stride = dst_stride[level];
  auto src_level_stride = src_stride[perm[level]];

  if (max_levels - level == Static) {
    transpose_impl_static<T, Static>(dst, src, max_levels, dst_stride, src_stride, size, perm);
  } else {
    for (int64_t i = 0; i < size[level]; i++) {
      transpose_impl<T, Static>(dst, src, level + 1, max_levels, dst_stride, src_stride, size,
                                perm);
      dst += dst_level_stride;
      src += src_level_stride;
    }
  }
}

/**
 * @brief Find blocks of consecutive numbers in the permutation that would be transposed together
 *
 * For example [0, 1, 5, 2, 3, 4] has following blocks: [[0, 1], [5], [2, 3, 4]]
 * and the result would be: {{0, 1}, {2, 2}, {3, 5}}
 *
 * @return List of starting and ending position of blocks(both ends inclusive)
 */
std::vector<std::pair<int, int>> PermutationBlocks(span<const int> perm) {
  if (perm.empty()) {
    return {};
  }
  std::vector<std::pair<int, int>> result;
  int current_start = 0;
  int current_end = 0;
  for (int i = 0; i < perm.size() - 1; i++) {
    if (perm[i] == perm[i + 1] - 1) {
      current_end = i + 1;
    } else {
      result.push_back({current_start, current_end});
      current_start = i + 1;
      current_end = i + 1;
    }
  }
  // We wouldn't add the last one in any case
  result.emplace_back(current_start, current_end);
  return result;
}

/**
 * @brief Based on the blocks returned from PermutationBlocks, collapse the blocks
 *        in the perm and re-enumerate them.
 *
 * @param perm Permutation to collapse
 * @param perm_blocks Description of consecutive blocks in perm
 * @return std::vector<int>
 */
std::vector<int> CollapsePermutation(span<const int> perm,
                                     const std::vector<std::pair<int, int>> &perm_blocks) {
  std::vector<int> result;
  std::vector<int> removed;
  result.reserve(perm_blocks.size());
  removed.reserve(perm.size() - perm_blocks.size());
  for (auto &block : perm_blocks) {
    result.push_back(perm[block.first]);
    for (int i = block.first + 1; i <= block.second; i++) {
      removed.push_back(perm[i]);
    }
  }
  for (auto &elem : result) {
    int original_elem = elem;
    for (auto removed_elem : removed) {
      if (original_elem > removed_elem) {
        elem--;
      }
    }
  }
  return result;
}

/**
 * @brief Based on the blocks returned from PermutationBlocks, collapse the blocks
 *        in the shape that are permuted together.
 *
 * @param shape Shape to be collapsed
 * @param perm Original, non-collapsed permutation
 * @param perm_blocks Description of blocks in permutation
 * @return TensorShape<>
 */
TensorShape<> CollapseShape(const TensorShape<> &shape, span<const int> perm,
                            const std::vector<std::pair<int, int>> &perm_blocks) {
  TensorShape<> result;
  result.resize(perm_blocks.size());
  auto collapsed_perm = CollapsePermutation(perm, perm_blocks);
  for (size_t i = 0; i < perm_blocks.size(); i++) {
    result[i] = 1;
  }
  for (size_t i = 0; i < perm_blocks.size(); i++) {
    int source_pos = collapsed_perm[i];
    for (int j = perm_blocks[i].first; j <= perm_blocks[i].second; j++) {
      result[source_pos] *= shape[perm[j]];
    }
  }
  assert(volume(result) == volume(shape));
  return result;
}

}  // namespace detail

/**
 * @brief Transpose `src` Tensor to `dst` wrt to permutation `perm`
 */
template <typename T>
void Transpose(const TensorView<StorageCPU, T> &dst, const TensorView<StorageCPU, const T> &src,
               span<const int> perm) {
  int N = src.shape.sample_dim();
  assert(dst.shape.sample_dim() == N);
  assert(volume(src.shape) == volume(dst.shape));
  auto dst_strides = GetStrides(dst.shape);
  auto src_strides = GetStrides(src.shape);
  VALUE_SWITCH(N, static_dims, (1, 2, 3), (
    detail::transpose_impl_static<T, static_dims, static_dims>(
        dst.data, src.data,
        static_dims, make_span(dst_strides), make_span(src_strides), dst.shape, perm);),
  (
    detail::transpose_impl<T, 3>(dst.data, src.data, 0, N,
        make_span(dst_strides), make_span(src_strides), dst.shape, perm);));
}

/**
 * @brief Transpose `src` Tensor to `dst` wrt to permutation `perm`
 *
 * Internally collapse the groups of consecutive dimensions for the transpositon
 *
 * For example "HWC", perm = {2, 0, 1}; the "HW" would be collapsed to one dimension "X",
 * and effectively we will do XC -> CX transposition.
 */
template <typename T>
void TransposeGrouped(const TensorView<StorageCPU, T> &dst,
                      const TensorView<StorageCPU, const T> &src, span<const int> perm) {
  int N = src.shape.sample_dim();
  auto src_shape = src.shape;
  auto perm_blocks = detail::PermutationBlocks(perm);
  auto collapsed_src_shape = detail::CollapseShape(src_shape, perm, perm_blocks);
  auto collapsed_perm = detail::CollapsePermutation(perm, perm_blocks);
  auto collapsed_dst_shape = Permute(collapsed_src_shape, collapsed_perm);
  Transpose(TensorView<StorageCPU, T>{dst.data, collapsed_dst_shape},
            TensorView<StorageCPU, const T>{src.data, collapsed_src_shape},
            make_cspan(collapsed_perm));
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_TRANSPOSE_H_
