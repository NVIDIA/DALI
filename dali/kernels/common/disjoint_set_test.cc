// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include "dali/kernels/common/disjoint_set.h"

#include "dali/core/format.h"

namespace dali {
namespace kernels {

template <typename T, typename Index, typename Ops, typename RNG>
void random_merge(disjoint_set<T, Index, Ops> ds,
                  T *data,
                  unsigned &mask,
                  RNG &&rng,
                  std::uniform_int_distribution<Index> &idx_dist,
                  std::bernoulli_distribution &op_order,
                  Index prev_idx = -1) {
  if (!mask)
    return;
  Index idx = idx_dist(rng);
  while ((mask & (1u<<idx)) == 0)
    idx = idx_dist(rng);
  mask &= ~(1u<<idx);

  bool merge_first = op_order(rng);

  if (merge_first) {
    if (prev_idx >= 0) {
      ds.merge(data, idx, prev_idx);
    }
  }
  random_merge(ds, data, mask, rng, idx_dist, op_order, idx);
  if (!merge_first) {
    if (prev_idx >= 0) {
      ds.merge(data, idx, prev_idx);
    }
  }
}

TEST(DisjointSet, Test1) {
  const int N = 32;
  int data[N];  // NOLINT
  disjoint_set<int> ds;
  ds.init(data);
  for (int i = 0; i < N; i++) {
    ASSERT_EQ(data[i], i);
  }

  std::mt19937_64 rng(12345);

  std::bernoulli_distribution op_order(0.5);
  std::uniform_int_distribution<> idx_dist(0, N-1);
  unsigned mask = 0xffffffffu;  // 32 bits set

  random_merge(ds, data, mask, rng, idx_dist, op_order);

  for (int j = 0; j < N; j++) {
    EXPECT_EQ(ds.find(data, j), 0);
  }

  for (int j = 0; j < N; j++) {
    EXPECT_EQ(data[j], 0);
  }
}

}  // namespace kernels
}  // namespace dali
