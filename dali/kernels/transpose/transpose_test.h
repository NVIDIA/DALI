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

#ifndef DALI_KERNELS_TRANSPOSE_TRANSPOSE_TEST_H_
#define DALI_KERNELS_TRANSPOSE_TRANSPOSE_TEST_H_

#include "dali/kernels/common/utils.h"

namespace dali {
namespace kernels {
namespace testing {

namespace {  // NOLINT(build/namespaces)
// All 4-element permutations
const int Permutations4[][4] = {
  { 0, 1, 2, 3 },
  { 0, 1, 3, 2 },
  { 0, 2, 1, 3 },
  { 0, 2, 3, 1 },
  { 0, 3, 1, 2 },
  { 0, 3, 2, 1 },
  { 1, 0, 2, 3 },
  { 1, 0, 3, 2 },
  { 1, 2, 0, 3 },
  { 1, 2, 3, 0 },
  { 1, 3, 0, 2 },
  { 1, 3, 2, 0 },
  { 2, 0, 1, 3 },
  { 2, 0, 3, 1 },
  { 2, 1, 0, 3 },
  { 2, 1, 3, 0 },
  { 2, 3, 0, 1 },
  { 2, 3, 1, 0 },
  { 3, 0, 1, 2 },
  { 3, 0, 2, 1 },
  { 3, 1, 0, 2 },
  { 3, 1, 2, 0 },
  { 3, 2, 0, 1 },
  { 3, 2, 1, 0 }
};
}  // namespace

template <typename T, typename Extent>
void RefTranspose(T *out, const uint64_t *out_strides,
                  const T *in, const uint64_t *in_strides, const Extent *shape, int ndim) {
  if (ndim == 0) {
    *out = *in;
  } else {
    for (Extent i = 0; i < *shape; i++) {
      RefTranspose(out, out_strides + 1, in, in_strides + 1, shape + 1, ndim - 1);
      out += *out_strides;
      in += *in_strides;
    }
  }
}

template <typename T, typename Extent>
void RefTranspose(T *out, const T *in, const Extent *in_shape, const int *perm, int ndim) {
  uint64_t out_strides[32], in_strides[32], tmp_strides[32], out_shape[32];
  CalcStrides(tmp_strides, in_shape, ndim);
  for (int i = 0; i < ndim; i++) {
    out_shape[i] = in_shape[perm[i]];
  }
  CalcStrides(out_strides, out_shape, ndim);
  for (int i = 0; i < ndim; i++) {
    in_strides[i] = tmp_strides[perm[i]];
  }

  RefTranspose(out, out_strides, in, in_strides, out_shape, ndim);
}

}  // namespace testing
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_TRANSPOSE_TRANSPOSE_TEST_H_
