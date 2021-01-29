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
#include "dali/core/util.h"
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
  // the mask is empty - all bits have been used
  if (!mask)
    return;
  Index idx = idx_dist(rng);
  while ((mask & (1u << idx)) == 0)  // if the index has been used (it's bit in the mask is 0)...
    idx = idx_dist(rng);             // ...generate a new index and retry
  mask &= ~(1u << idx);

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

template <typename Sequence>
void CheckNoForwardLinks(Sequence &&seq) {
  auto b = dali::begin(seq);
  auto e = dali::end(seq);
  auto i = b;
  auto index = *b;
  for (++i; i != e; ++i, ++index) {
    if (*i < index) {
      std::stringstream msg;
      msg << "The group index cannot point to an element further on the list; got: ";
      for (auto x : seq)
        msg << " " << x;
      EXPECT_GE(*i, *b) << msg.str();
    }
  }
}

TEST(DisjointSet, BasicTest) {
  const int N = 10;
  int data[N];  // NOLINT
  disjoint_set<int> ds;
  ds.init(data);

  for (int i = 0; i < N; i++) {
    ASSERT_EQ(data[i], i);
    ASSERT_EQ(ds.find(data, i), i);
  }

  ds.merge(data, 0, 1);
  CheckNoForwardLinks(data);
  EXPECT_EQ(ds.find(data, 0), 0);
  EXPECT_EQ(ds.find(data, 1), 0);

  ds.merge(data, 3, 2);
  CheckNoForwardLinks(data);
  EXPECT_EQ(ds.find(data, 2), 2);
  EXPECT_EQ(ds.find(data, 3), 2);

  ds.merge(data, 6, 5);
  CheckNoForwardLinks(data);
  EXPECT_EQ(data[6], 5);
  ds.merge(data, 4, 5);
  CheckNoForwardLinks(data);
  EXPECT_EQ(data[5], 4);
  EXPECT_EQ(data[6], 5);

  ds.merge(data, 4, 0);
  CheckNoForwardLinks(data);
  EXPECT_EQ(data[4], 0);
  EXPECT_EQ(ds.find(data, 6), 0);
  EXPECT_EQ(data[6], 0) << "`find` should update the entry.";

  ds.merge(data, 8, 9);
  CheckNoForwardLinks(data);
  ds.merge(data, 7, 9);
  CheckNoForwardLinks(data);
  EXPECT_EQ(ds.find(data, 8), 7);
  EXPECT_EQ(data[8], 7) << "`find` should update the entry.";
  ds.merge(data, 6, 7);
  CheckNoForwardLinks(data);
  EXPECT_EQ(ds.find(data, 9), 0) << "`merge` didn't propagate";
  ds.merge(data, 8, 3);
  CheckNoForwardLinks(data);
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(ds.find(data, i), 0);
    EXPECT_EQ(data[i], 0) << "`find` should update the entry.";
  }
}

TEST(DisjointSet, RandomMergeAll) {
  const int N = 32;
  int data[N];  // NOLINT
  disjoint_set<int> ds;
  std::mt19937_64 rng(12345);

  for (int iter = 0; iter < 10; iter++) {
    ds.init(data);
    for (int i = 0; i < N; i++) {
      ASSERT_EQ(data[i], i);
    }

    std::bernoulli_distribution op_order(0.5);
    std::uniform_int_distribution<> idx_dist(0, N-1);
    unsigned mask = 0xffffffffu;  // 1 bit for each element of the array - initially, all set

    random_merge(ds, data, mask, rng, idx_dist, op_order);

    for (int j = 0; j < N; j++) {
      EXPECT_EQ(ds.find(data, j), 0);
    }

    for (int j = 0; j < N; j++) {
      EXPECT_EQ(data[j], 0);
    }
  }
}

}  // namespace kernels
}  // namespace dali
