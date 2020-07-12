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

#ifndef DALI_KERNELS_TRANSPOSE_TRANSPOSE_H_
#define DALI_KERNELS_TRANSPOSE_TRANSPOSE_H_

#include <type_traits>
#include <utility>
#include <vector>

#include "dali/core/static_switch.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/transpose/transpose_util.h"

namespace dali {
namespace kernels {
namespace transpose_impl {

/**
 * @brief Transpose recursion that should allow to inline innermost loops
 *        The innermost case where the data is actually copied.
 *
 * @tparam LevelsLeft how many levels of recursion are left, here equal to 1
 * @tparam MaxLevels statically known number of dimensions or -1 otherwise
 */
template <int LevelsLeft, int MaxLevels = -1, typename T>
std::enable_if_t<LevelsLeft == 1> TransposeImplStatic(T *dst, const T *src, int max_levels,
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

/**
 * @brief Transpose recursion that should allow to inline innermost loops.
 *
 * @tparam LevelsLeft how many levels of recursion are left, here grater than 1
 * @tparam MaxLevels statically known number of dimensions or -1 otherwise
 */
template <int LevelsLeft, int MaxLevels = -1, typename T>
std::enable_if_t<(LevelsLeft > 1)> TransposeImplStatic(T *dst, const T *src, int max_levels,
                                                       span<const int64_t> dst_stride,
                                                       span<const int64_t> src_stride,
                                                       TensorShape<> size, span<const int> perm) {
  const auto level = MaxLevels >= 0 ? (MaxLevels - LevelsLeft) : max_levels - LevelsLeft;
  auto dst_level_stride = dst_stride[level];
  auto src_level_stride = src_stride[perm[level]];
  for (int64_t i = 0; i < size[level]; i++) {
    TransposeImplStatic<LevelsLeft - 1, MaxLevels>(dst, src, max_levels, dst_stride, src_stride,
                                                      size, perm);
    dst += dst_level_stride;
    src += src_level_stride;
  }
}

/**
 * @brief Generic case for transpose, that transposes the tensors.
 *
 * Goes over the data in the dst-order, assumes at least one element and that
 * Static <= max_levels - level.
 *
 * @tparam Static how many innermost dimensions should use template/static recursion.
 *         Must be at least 1.
 * @tparam T type of the data
 * @param dst pointer to
 * @param src
 * @param level recursion level
 * @param max_levels number of dimensions/recurion levels
 * @param dst_stride
 * @param src_stride
 * @param size dst-ordered shape
 * @param perm target permutation, source dimension `perm[i]` goes to destination dimension `i`
 */
template <int Static = 3, typename T>
void TransposeImpl(T *dst, const T *src, int level, int max_levels, span<const int64_t> dst_stride,
                   span<const int64_t> src_stride, TensorShape<> size, span<const int> perm) {
  static_assert(Static > 0, "At least last dimension must be `unrolled` so the data can be copied");
  assert(max_levels - level >= Static &&
         "The implementation cannot execute template recursion of requested level `Static` for "
         "specified `max_levels` of recursion from current starting `level`.");
  auto dst_level_stride = dst_stride[level];
  auto src_level_stride = src_stride[perm[level]];

  if (max_levels - level == Static) {
    TransposeImplStatic<Static>(dst, src, max_levels, dst_stride, src_stride, size, perm);
  } else {
    for (int64_t i = 0; i < size[level]; i++) {
      TransposeImpl<Static>(dst, src, level + 1, max_levels, dst_stride, src_stride, size, perm);
      dst += dst_level_stride;
      src += src_level_stride;
    }
  }
}

}  // namespace transpose_impl

/**
 * @brief Transpose `src` Tensor to `dst` wrt to permutation `perm`
 *
 * Source dimension `perm[i]` goes to destination dimension `i`.
 */
template <typename T>
void Transpose(const TensorView<StorageCPU, T> &dst, const TensorView<StorageCPU, const T> &src,
               span<const int> perm) {
  int N = src.shape.sample_dim();
  if (N == 0) {  // it's a scalar - just copy it
    *dst.data = *src.data;
    return;
  }
  assert(dst.shape.sample_dim() == N);
  assert(volume(src.shape) == volume(dst.shape));
  auto dst_strides = GetStrides(dst.shape);
  auto src_strides = GetStrides(src.shape);
  VALUE_SWITCH(N, static_dims, (1, 2, 3), (
    transpose_impl::TransposeImplStatic<static_dims, static_dims>(
        dst.data, src.data,
        static_dims, make_span(dst_strides), make_span(src_strides), dst.shape, perm);),
  (
    transpose_impl::TransposeImpl(dst.data, src.data, 0, N,
        make_span(dst_strides), make_span(src_strides), dst.shape, perm);));
}

/**
 * @brief Transpose `src` Tensor to `dst` wrt to permutation `perm`
 *
 * Internally collapse the groups of consecutive dimensions for the transposition
 *
 * For example "HWC", perm = {2, 0, 1}; the "HW" would be collapsed to one dimension "X",
 * and effectively we will do XC -> CX transposition.
 *
 * Source dimension `perm[i]` goes to destination dimension `i`.
 */
template <typename T>
void TransposeGrouped(const TensorView<StorageCPU, T> &dst,
                      const TensorView<StorageCPU, const T> &src, span<const int> perm) {
  int N = src.shape.sample_dim();
  auto src_shape = src.shape;
  TensorShape<> collapsed_src_shape;
  SmallVector<int, DynamicTensorShapeContainer::static_size> collapsed_perm;
  transpose_impl::SimplifyPermute(collapsed_src_shape, collapsed_perm, src.shape, perm);
  auto collapsed_dst_shape = permute(collapsed_src_shape, collapsed_perm);
  Transpose(TensorView<StorageCPU, T>{dst.data, collapsed_dst_shape},
            TensorView<StorageCPU, const T>{src.data, collapsed_src_shape},
            make_cspan(collapsed_perm));
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_TRANSPOSE_TRANSPOSE_H_
