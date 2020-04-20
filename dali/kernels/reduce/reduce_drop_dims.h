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

#ifndef DALI_KERNELS_REDUCE_REDUCE_DROP_DIMS_H_
#define DALI_KERNELS_REDUCE_REDUCE_DROP_DIMS_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include "dali/core/util.h"
#include "dali/core/host_dev.h"

namespace dali {
namespace kernels {
namespace reduce_impl {

struct DropDims {
  static constexpr int kMaxDims = 4;

  int64_t div[kMaxDims];
  int64_t mul[kMaxDims];
  int64_t mod[kMaxDims];
  int start = 2 * kMaxDims;

  DropDims() = default;

  template <typename Indices>
  static int simplify(int64_t *out_shape, unsigned &out_mask,
                      const Indices &in_shape, unsigned axis_mask) {
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
  DropDims(const Indices &in_shape, unsigned axis_mask) {
    int64_t shape[kMaxDims];
    int d = simplify(shape, axis_mask, in_shape, axis_mask);
    start = 2 * kMaxDims;

    if (d == 1) {
      if (axis_mask == 1)
        start = -1;
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
      if ((axis_mask & (1 << i)) == 0) {
        vol_kept *= shape[i];
        kept_dims++;
      } else {
        reduced_dims++;
      }
    }

    bool mod_first = (axis_mask & 1);

    for (int i = 0; i < d; i++) {
      if (volumes[i] == 1)
        break;
      if (axis_mask & (1 << i)) {
        mod[nmod++] = volumes[i];
        continue;
      }
      div[ndiv] = volumes[i];
      mul[ndiv] = kept_volumes[i];
      ndiv++;
    }

    assert(abs(ndiv - nmod) <= 1);

    // Now we move the divisors/moduli to the end of the arrays, so we can use the unrolled loop
    // with fixed indices.

    int div_ofs = !mod_first && ndiv < nmod ? 1 : 0;
    int mod_ofs = nmod < ndiv || (nmod == ndiv && mod_first) ? 1 : 0;

    if (ndiv + div_ofs > kMaxDims)
      throw std::out_of_range("Maximum number of dimension groups exceeded");

    if (nmod + mod_ofs > kMaxDims)
      throw std::out_of_range("Maximum number of dimension groups exceeded");

    if (div_ofs) {
      div[kMaxDims - 1] = 1;
      mul[kMaxDims - 1] = 0;
    }
    if (mod_ofs || !nmod)
      mod[kMaxDims - 1] = 1;
    for (int i = ndiv-1; i >= 0; i--) {
      div[kMaxDims - ndiv + i - div_ofs] = div[i];
      mul[kMaxDims - ndiv + i - div_ofs] = mul[i];
    }
    for (int i = nmod-1; i >= 0; i--) {
      mod[kMaxDims - nmod + i - mod_ofs] = mod[i];
    }

    start = std::min(2*(kMaxDims - ndiv - div_ofs), 2*(kMaxDims - nmod - mod_ofs)) + mod_first;
  }

  DALI_HOST_DEV int64_t reindex(int64_t index) const {
    int64_t out = 0;

    // This is an unrolled loop over dimensions.
    // We can start either at modulo or at division, depending on whether
    // the outermost dimension is reduced or not.
    switch (start) {
      case -1:
        return 0;  // special case - reduced down to a scalar!

    // Warning: intentional fall-through!
    #define REINDEX_CASE(idx)\
      case 2*idx:\
        out += index / div[idx] * mul[idx];\
      case 2*idx+1:\
        if (mod[idx] == 1) {\
          index = 0;\
          break;\
        }\
        index = index % mod[idx]

      REINDEX_CASE(0);
      REINDEX_CASE(1);
      REINDEX_CASE(2);
      REINDEX_CASE(3);
    }

    out += index;

    return out;
  }
};
}  // namespace reduce_impl
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_REDUCE_REDUCE_DROP_DIMS_H_
