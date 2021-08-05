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

#ifndef DALI_KERNELS_REDUCE_REDUCE_DROP_DIMS_H_
#define DALI_KERNELS_REDUCE_REDUCE_DROP_DIMS_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include "dali/core/util.h"
#include "dali/core/cuda_utils.h"
#include "dali/core/host_dev.h"
#include "dali/core/fast_div.h"

namespace dali {
namespace kernels {
namespace reduce_impl {

/**
 * Calculates a flat index in a tensor with reduced dimensions based on the
 * index in original tensor.
 *
 * This is used in calculation of standard deviation when subtracting mean.
 *
 * @details When reducing non-adjacent dimensions, the outer reduction will see
 * all inner dimensions flattened - reduced and non-reduced alike. This flat index
 * cannot be used to index the the tensor of means, which has more reduced dimensions.
 * Example - calculating mean/stddev of all pixel brightness values in a mutli-channel video:
 * input layout FHWC
 * output layout 1HW1
 * When calculating variance in F, we still see HWC as the (fused) inner dimension, but it is
 * different than HW1 in the tensor of means. This class implements this kind of reindexing.
 *
 * The reindexing is done by either dividing and multiplying by old/new strides or by taking modulo.
 */
template <int max_dims>
struct DropDims {
  static constexpr int kMaxDims = max_dims;

  // improved packing, divisors not stored for div (only for modulo)
  uint64_t div_m[kMaxDims];
  uint64_t mod_m[kMaxDims];
  uint64_t mod_d[kMaxDims];
  uint8_t div_add[kMaxDims], div_shift[kMaxDims];
  uint8_t mod_add[kMaxDims], mod_shift[kMaxDims];
  int64_t mul_m[kMaxDims];
  int start = 2 * kMaxDims;

  DALI_HOST_DEV DALI_FORCEINLINE fast_div<uint64_t> div(int idx) const {
    fast_div<uint64_t> fd;
    fd.divisor = 0;
    fd.mul = div_m[idx];
    fd.add = div_add[idx];
    fd.shift = div_shift[idx];
    return fd;
  }

  DALI_HOST_DEV DALI_FORCEINLINE fast_div<uint64_t> mod(int idx) const {
    fast_div<uint64_t> fd;
    fd.divisor = mod_d[idx];
    fd.mul = mod_m[idx];
    fd.add = mod_add[idx];
    fd.shift = mod_shift[idx];
    return fd;
  }

  DALI_HOST_DEV DALI_FORCEINLINE int64_t mul(int idx) const {
    return mul_m[idx];
  }

  DALI_HOST_DEV DALI_FORCEINLINE void div(int idx, fast_div<uint64_t> v) {
    div_m[idx] = v.mul;
    div_add[idx] = v.add;
    div_shift[idx] = v.shift;
  }

  DALI_HOST_DEV DALI_FORCEINLINE void mod(int idx, fast_div<uint64_t> v) {
    mod_d[idx] = v.divisor;
    mod_m[idx] = v.mul;
    mod_add[idx] = v.add;
    mod_shift[idx] = v.shift;
  }

  DALI_HOST_DEV DALI_FORCEINLINE void mul(int idx, int64_t m) {
    mul_m[idx] = m;
  }

  DALI_HOST_DEV DropDims() {}

  /**
   * Collapses adjacent groups of reduced/non-reduced dimensions.
   * The input can have up to 64 dimensions, however, the result must have
   * at most 8 groups of reduced/non-reduced dimensions.
   * Dimensions with unit extent can be collapsed with either neighbor.
   *
   * @param out_shape   simplified shape
   * @param out_mask    simplified mask (alternating bit pattern)
   * @param in_shape    original shape
   * @param axis_mask   mask, where 1 at position i indicates that the i-th dimension is reduced
   * @return number of dimensions after simplification
   *
   * Example 1:
   * ```
   * in_shape:  [ 2, 3, 4, 5, 6 ]
   * reduced:     ^        ^  ^
   * axis_mask = 0b11001  (looks reversed - LSB is dim 0)
   *
   * out_shape: [ 2, 12, 30 ]
   * reduced:     ^       ^
   * out_mask = 0b101
   * ```
   *
   * Example 2 (collapsing unit dim):
   * ```
   * in_shape:  [ 2, 1, 3, 4, 5 ]
   * reduced:     ^     ^  ^
   * axis_mask = 0b01101  (looks reversed - LSB is dim 0)
   *
   * out_shape: [ 24, 5 ]
   * reduced:     ^
   * out_mask = 0b10
   * ```
   */
  template <typename Indices>
  static int simplify(int64_t *out_shape, unsigned &out_mask,
                      const Indices &in_shape, uint64_t axis_mask) {
    int dims = size(in_shape);
    int d = 0;
    out_shape[0] = in_shape[0];
    bool prev = axis_mask & 1;
    out_mask = prev ? 1u : 0u;
    for (int i = 1; i < dims; i++) {
      if (in_shape[i] == 1)
        continue;
      bool flag = (axis_mask >> i) & 1;
      if (flag != prev) {
        d++;
        if (d > 2*kMaxDims)
          throw std::out_of_range("Maximum number of dimension groups exceeded");
        out_shape[d] = in_shape[i];
        out_mask |= (flag ? 1u : 0u) << d;
      } else {
        out_shape[d] *= in_shape[i];
      }
      prev = flag;
    }
    d++;
    if (d > 2*kMaxDims)
      throw std::out_of_range("Maximum number of dimension groups exceeded");

    return d;
  }

  /**
   * @brief Initializes reindexing given shape and mask.
   */
  template <typename Indices>
  DropDims(const Indices &in_shape, uint64_t reduced_axes) {
    memset(this, 0, sizeof(*this));
    if (size(in_shape) == 0) {
      start = -1;
      return;
    }
    int64_t shape[2*kMaxDims];
    unsigned axis_mask;
    int d = simplify(shape, axis_mask, in_shape, reduced_axes);

    if (d == 1) {
      // short circuit trivial case
      if (axis_mask == 1)
        start = -1;
      else
        start = 2 * kMaxDims;
      return;
    }

    int nmod = 0;
    int ndiv = 0;

    int64_t volumes[2*kMaxDims];
    int64_t kept_volumes[2*kMaxDims];
    int64_t vol_total = 1, vol_kept = 1;
    int reduced_dims = 0;
    int kept_dims = 0;
    for (int i = d - 1; i >= 0; i--) {
      volumes[i] = vol_total;
      kept_volumes[i] = vol_kept;
      vol_total *= shape[i];
      if ((axis_mask & (1u << i)) == 0) {
        vol_kept *= shape[i];
        kept_dims++;
      } else {
        reduced_dims++;
      }
    }

    bool mod_first = (axis_mask & 1);

    // If a dimension is kept, the index is calculated by dividing by original subvolume
    // and multiplying by the new subvolume.
    // If a dimension is dropped, the index is taken modulo subvolume.

    for (int i = 0; i < d - 1; i++) {
      assert(volumes[i] > 1);  // simplification should make this impossible
      if (axis_mask & (1u << i)) {
        mod(nmod++, volumes[i]);
      } else {
        div(ndiv, volumes[i]);
        mul(ndiv, kept_volumes[i]);
        ndiv++;
      }
    }

    assert(std::abs(ndiv - nmod) <= 1);
    assert(mod_first || ndiv >= nmod);

    // Now we move the divisors/moduli to the end of the arrays, so we can use the unrolled loop
    // with fixed indices.
    // See `reindex` function for examples.

    int mod_ofs = nmod < ndiv || (nmod == ndiv && mod_first) ? 1 : 0;

    if (ndiv > kMaxDims)
      throw std::out_of_range("Maximum number of dimension groups exceeded");

    if (nmod + mod_ofs > kMaxDims)
      throw std::out_of_range("Maximum number of dimension groups exceeded");

    if (mod_ofs || !nmod)
      mod(kMaxDims - 1, 1);  // pad with no-op mod

    for (int i = ndiv-1; i >= 0; i--) {
      div(kMaxDims - ndiv + i, div(i));
      mul(kMaxDims - ndiv + i, mul(i));
    }
    for (int i = nmod-1; i >= 0; i--) {
      mod(kMaxDims - nmod + i - mod_ofs, mod(i));
    }

    // start index - even if starting with div/mul, odd if starting with mod
    start = mod_first
              ? 2 * (kMaxDims - nmod - mod_ofs) + 1
              : 2 * (kMaxDims - ndiv);
  }

  DALI_HOST_DEV int64_t reindex(int64_t index) const {
    int64_t out = 0;

    // Now we run through the divmul/mod stages. Since we've simplified the problem,
    // the stages appear in an alternating pattern.
    // The start can be at either divmul or mod, depending on `start` parity.
    // There's also a special value of -1 which denotes full reduction - just return index 0.

    // Example - 2 divs, 2 mods, mod_first = true (outermost dimension is reduced)
    // D - div/mul stage
    // M - mod stage
    //
    // D 0   ---
    // M 0   ---
    // D 1   ---
    // M 1   mod 0           <-- start = 3
    // D 2   div 0
    // M 2   mod 1
    // D 3   div 1
    // M 3   1 (no-op)

    // If a modulus is equal to 1, the operation is skipped - `if` is faster than integer division.

    // This is an unrolled loop over dimensions.
    // We can start either at modulo or at division, depending on whether
    // the outermost dimension is reduced or not.
    switch (start) {
      case -1:
        return 0;  // special case - reduced down to a scalar!

    // Warning: intentional fall-through!
    #define REINDEX_CASE(idx)\
      case 2*idx:\
        if (idx < kMaxDims)\
          out += static_cast<uint64_t>(index) / div(idx) * mul(idx);\
      case 2*idx+1:\
        if (idx < kMaxDims) {\
          if (mod(idx) == 1) {\
            index = 0;\
            break;\
          }\
          index = static_cast<uint64_t>(index) % mod(idx);\
        }

      REINDEX_CASE(0)
      REINDEX_CASE(1)
      REINDEX_CASE(2)
      REINDEX_CASE(3)
      REINDEX_CASE(4)
      static_assert(max_dims <= 5, "Add more switch cases for max_dims > 5");
    }

    out += index;

    return out;
  }
};

}  // namespace reduce_impl
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_REDUCE_REDUCE_DROP_DIMS_H_
