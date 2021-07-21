// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <random>
#include "dali/core/bitmask.h"

namespace dali {
namespace test {

TEST(Bitmask, ResizeFullWord) {
  for (bool value : { false, true }) {
    bitmask mask;
    const size_t size = 3 * bitmask::storage_bits;
    mask.resize(size, value);
    EXPECT_EQ(mask.size(), size);
    for (size_t i = 0; i < size; i++)
      EXPECT_EQ(mask[i], value);
  }
}

TEST(Bitmask, ResizePartialWord) {
  bitmask mask;
  mask.resize(3, false);
  mask.resize(7, true);
  EXPECT_EQ(mask.data()[0], bitmask::bit_storage_t(0b1111000));
  mask.resize(6, true);
  EXPECT_EQ(mask.data()[0], bitmask::bit_storage_t(0b111000));
  mask.resize(5, true);
  EXPECT_EQ(mask.data()[0], bitmask::bit_storage_t(0b11000));
}

TEST(Bitmask, Indexing) {
  bitmask mask;
  mask.resize(100, false);
  mask[1] = true;
  mask[2] = true;
  mask[3] = true;
  mask[2] = false;
  mask[99] = true;
  EXPECT_FALSE(mask[0]);
  EXPECT_TRUE(mask[1]);
  EXPECT_FALSE(mask[2]);
  EXPECT_TRUE(mask[3]);

  EXPECT_TRUE(mask[99]);
  mask[99] ^= false;
  EXPECT_TRUE(mask[99]);
  mask[99] ^= true;
  EXPECT_FALSE(mask[99]);
  mask[99] ^= false;
  EXPECT_FALSE(mask[99]);

  mask[99] |= false;
  EXPECT_FALSE(mask[99]);
  mask[99] |= true;
  EXPECT_TRUE(mask[99]);
  mask[99] |= false;
  EXPECT_TRUE(mask[99]);

  mask[99] &= true;
  EXPECT_TRUE(mask[99]);
  mask[99] &= false;
  EXPECT_FALSE(mask[99]);
  mask[99] &= true;
  EXPECT_FALSE(mask[99]);

  EXPECT_EQ(mask.data()[0], bitmask::bit_storage_t(0b1010));
}

TEST(Bitmask, Fill) {
  bitmask mask;
  mask.resize(500, false);
  mask.fill(3, 5, true);
  EXPECT_EQ(mask.data()[0], bitmask::bit_storage_t(0b11000));
  mask.fill(0, 10, true);
  EXPECT_EQ(mask.data()[0], bitmask::bit_storage_t(0b1111111111));
  mask.fill(4, 8, false);
  EXPECT_EQ(mask.data()[0], bitmask::bit_storage_t(0b1100001111));

  for (int start  : { 0, 64, 9 }) {
    for (int end    : { 7, 64, 67, 128, 199 }) {
      for (bool value : { false, true }) {
        std::cerr << "Filling range " << start << " to " << end << " with " << (value ? 1 : 0)
                  << std::endl;
        mask.fill(!value);
        mask.fill(start, end, value);
        for (int i = 0; i < 500; i++) {
          bool expected = (i >= start && i < end) ? value : !value;
          ASSERT_EQ(mask[i], expected) << " @ index " << i;
        }
      }
    }
  }
}

TEST(Bitmask, PushPop) {
  bitmask mask;
  std::vector<bool> ref;
  std::mt19937_64 rng(12345);
  std::bernoulli_distribution what;
  std::bernoulli_distribution action(0.8);
  for (int i = 0; i < 1000; i++) {
    if (action(rng) || ref.empty()) {
      bool value = what(rng);
      mask.push_back(value);
      ref.push_back(value);
    } else {
      mask.pop_back();
      ref.pop_back();
    }
    EXPECT_EQ(mask.size(), ref.size());
  }

  for (int i = 0; i < static_cast<int>(ref.size()); i++) {
    EXPECT_EQ(mask[i], ref[i]);
  }
}

TEST(Bitmask, Find) {
  bitmask mask;
  std::vector<bool> ref;
  std::mt19937_64 rng(12345);
  std::bernoulli_distribution what;
  std::uniform_int_distribution<> len_dist(1, 200);
  for (int i = 0; i < 200; i++) {
    int rep = len_dist(rng);
    bool value = what(rng);
    for (int j = 0; j < rep; j++) {
      mask.push_back(value);
      ref.push_back(value);
    }
  }

  for (bool value : { false, true }) {
    for (size_t start = 0; start < ref.size(); start++) {
      ptrdiff_t idx = mask.find(value, start);
      ptrdiff_t ref_idx = std::find(ref.begin() + start, ref.end(), value) - ref.begin();
      EXPECT_EQ(idx, ref_idx);
    }
  }
}

void TestBitmaskAppend(int first_bits, int second_bits) {
  bitmask m1;
  bitmask m2;
  std::vector<bool> ref;

  std::mt19937_64 rng(12345);
  std::bernoulli_distribution dist;
  int N = first_bits + second_bits;
  for (int i = 0; i < N; i++) {
    bool value = dist(rng);
    (i < first_bits ? m1 : m2).push_back(value);
    ref.push_back(value);
  }
  m1.append(m2);
  EXPECT_EQ(m1.ssize(), N);
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(m1[i], ref[i]);
  }
}

TEST(Bitmask, AppendToAligned) {
  TestBitmaskAppend(128, 1);
  TestBitmaskAppend(64, 63);
  TestBitmaskAppend(192, 128);
  TestBitmaskAppend(128, 65);
}

TEST(Bitmask, AppendNoOverflow) {
  TestBitmaskAppend(17, 13);
  TestBitmaskAppend(17, 64 - 17);
  TestBitmaskAppend(129 + 17, 64  +13);
  TestBitmaskAppend(64 + 17, 64  +13);
}

TEST(Bitmask, AppendOverflow) {
  TestBitmaskAppend(17, 63);
  TestBitmaskAppend(63, 2);
  TestBitmaskAppend(128 - 5, 9);
  TestBitmaskAppend(128 - 5, 128 + 9);
}

}  // namespace test
}  // namespace dali
