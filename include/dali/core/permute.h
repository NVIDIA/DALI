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

#ifndef DALI_CORE_PERMUTE_H_
#define DALI_CORE_PERMUTE_H_

#include "dali/core/host_dev.h"
#include "dali/core/util.h"
#include "dali/core/traits.h"

namespace dali {

DALI_NO_EXEC_CHECK
template <typename OutContainer, typename InContainer, typename Permutation>
DALI_HOST_DEV
void permute(OutContainer &&out, const InContainer &in, const Permutation &source_indices) {
  int n = dali::size(in);
  resize_if_possible(out, n);
  for (int d = 0; d < n; d++) {
    out[d] = in[source_indices[d]];
  }
}


DALI_NO_EXEC_CHECK
template <typename Container, typename Permutation>
DALI_HOST_DEV
Container permute(const Container &container, const Permutation &source_indices) {
  Container permuted_container;
  permute(permuted_container, container, source_indices);
  return permuted_container;
}

DALI_NO_EXEC_CHECK
template <typename OutContainer, typename InContainer, typename Permutation>
DALI_HOST_DEV
OutContainer permute(const InContainer &container, const Permutation &source_indices) {
  OutContainer permuted_container;
  permute(permuted_container, container, source_indices);
  return permuted_container;
}

DALI_NO_EXEC_CHECK
template <typename OutPerm, typename InPerm>
DALI_HOST_DEV
void inverse_permutation(OutPerm &&inv_perm, const InPerm &permutation) {
  int n = dali::size(permutation);
  resize_if_possible(inv_perm, n);
  for (int d = 0; d < n; d++) {
    auto perm_d = permutation[d];
    inv_perm[perm_d] = d;
  }
}

DALI_NO_EXEC_CHECK
template <typename Permutation>
DALI_HOST_DEV
Permutation inverse_permutation(const Permutation &permutation) {
  Permutation inv_perm;
  inverse_permutation(inv_perm, permutation);
  return inv_perm;
}

DALI_NO_EXEC_CHECK
template <typename OutPerm, typename InPerm>
DALI_HOST_DEV
OutPerm inverse_permutation(const InPerm &permutation) {
  OutPerm inv_perm;
  inverse_permutation(inv_perm, permutation);
  return inv_perm;
}

}  // namespace dali

#endif  // DALI_CORE_PERMUTE_H_
