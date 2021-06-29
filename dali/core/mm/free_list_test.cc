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

#include <gtest/gtest.h>
#include <random>
#include "dali/core/mm/detail/free_list.h"

namespace dali {
namespace mm {
namespace test {

TEST(MMUniformFreeList, PutGet) {
  uniform_free_list fl;
  char a[2];
  fl.put(a);
  EXPECT_EQ(fl.get(), static_cast<void*>(a));
  EXPECT_EQ(fl.get(), nullptr);
  fl.put(a);
  fl.put(a+1);
  EXPECT_EQ(fl.get(), static_cast<void*>(a+1));
  EXPECT_EQ(fl.get(), static_cast<void*>(a+0));
  EXPECT_EQ(fl.get(), nullptr);
}

TEST(MMUniformFreeList, PutMoveGet) {
  uniform_free_list l1, l2;
  char a[2];
  l1.put(a);
  l1.put(a+1);
  l2 = std::move(l1);
  EXPECT_EQ(l1.get(), nullptr) << "Should be empty - it's been moved";
  EXPECT_EQ(l2.get(), a+1);
  EXPECT_EQ(l2.get(), a);
  EXPECT_EQ(l2.get(), nullptr);
}

TEST(MMBestFitFreeList, PutGet) {
  best_fit_free_list fl;
  char a alignas(16)[1000];
  fl.put(a, 10);
  fl.put(a+10, 100);
  fl.put(a+110, 15);
  fl.put(a+125, 35);
  EXPECT_EQ(fl.get(10, 16), a);
  fl.put(a, 10);
  EXPECT_EQ(fl.get(8, 16), a);
  EXPECT_EQ(fl.get(2, 1), a+8);
  fl.put(a, 8);
  EXPECT_EQ(fl.get(100, 16), nullptr);
  EXPECT_EQ(fl.get(100, 2), a+10);
  EXPECT_EQ(fl.get(8, 16), a);
  EXPECT_EQ(fl.get(9, 16), a+112);
}


TEST(MMBestFitFreeList, PutGetMoveGet) {
  best_fit_free_list l1, l2;
  char a alignas(16)[1000];
  l1.put(a, 10);
  l1.put(a+10, 100);
  l1.put(a+110, 15);
  l1.put(a+125, 35);
  EXPECT_EQ(l1.get(11, 1), a+110);
  l2 = std::move(l1);
  EXPECT_EQ(l1.get(1, 1), nullptr) << "Should be empty - it's been moved";
  EXPECT_NE(l1.get(11, 1), a+110) << "This entry was removed in the original list, before move.";
  EXPECT_EQ(l2.get(100, 16), nullptr);
  EXPECT_EQ(l2.get(100, 2), a+10);
}


template <typename FreeList>
void TestCoalescingRemoveIf() {
  FreeList fl;
  char storage alignas(16)[4000];
  char *a = storage + 128;   // thank you, overzealous compiler!
  ASSERT_FALSE(fl.remove_if_in_list(a, 1));  // total removed: 250..750
  fl.put(a, 500);
  fl.put(a + 500, 500);
  ASSERT_FALSE(fl.remove_if_in_list(a + 2000, 1));  // totally outside
  ASSERT_FALSE(fl.remove_if_in_list(a - 100, 10));

  ASSERT_FALSE(fl.remove_if_in_list(a + 1000, 1));  // edge case
  ASSERT_FALSE(fl.remove_if_in_list(a - 100, 100));

  ASSERT_FALSE(fl.remove_if_in_list(a + 500, 501));  // overlaps, but exceeds
  ASSERT_FALSE(fl.remove_if_in_list(a - 1, 100));

  ASSERT_TRUE(fl.remove_if_in_list(a + 250, 500));  // total removed: 250..750

  ASSERT_FALSE(fl.remove_if_in_list(a + 250, 500));  // already removed
  ASSERT_FALSE(fl.remove_if_in_list(a + 300, 500));  // overlaps removed piece
  ASSERT_FALSE(fl.remove_if_in_list(a + 200, 500));

  ASSERT_TRUE(fl.remove_if_in_list(a + 200, 50));  // total removed: 200..750
  ASSERT_TRUE(fl.remove_if_in_list(a, 50));  // total removed: 0..50, 200..750
  EXPECT_EQ(fl.get(150, 1), a + 50);  // there's a gap at 50..200
  EXPECT_EQ(fl.get(250, 128), nullptr);  // there's a block large enough, but not aligned to 128
  EXPECT_EQ(fl.get(250, 1), a + 750);
  EXPECT_EQ(fl.get(1, 1), nullptr);  // the free list should be empty now

  // reset the free list
  fl.put(a, 1000);
  ASSERT_TRUE(fl.remove_if_in_list(a + 250, 500));  // total removed: 250..750
  fl.put(a + 250, 500);  // put back
  EXPECT_EQ(fl.get(1000, 1), a);
  EXPECT_EQ(fl.get(1, 1), nullptr);  // the free list should be empty now
}

template <typename FreeList>
void TestCoalescingPutGet() {
  FreeList fl;
  char a alignas(16)[1000];
  // put some pieces and let the list coalesce
  fl.put(a, 10);
  fl.put(a+10, 100);
  fl.put(a+110, 15);
  fl.put(a+125, 35);
  // check if we can get a contiguous block
  EXPECT_EQ(fl.get(160, 16), a);
  // put it back in pieces
  fl.put(a+125, 35);
  fl.put(a+10, 100);
  fl.put(a, 10);
  fl.put(a+110, 15);

  // now some random stuff
  EXPECT_EQ(fl.get(10, 16), a);
  fl.put(a, 10);
  EXPECT_EQ(fl.get(8, 16), a);
  EXPECT_EQ(fl.get(2, 1), a+8);
  fl.put(a, 8);
  EXPECT_EQ(fl.get(8, 16), a);
  EXPECT_EQ(fl.get(100, 8), a+16);
  fl.put(a, 8);
  fl.put(a+8, 2);
  EXPECT_EQ(fl.get(9, 16), a);

  // put everything back again
  fl.put(a+16, 100);
  fl.put(a, 9);
  // and check coalescing
  EXPECT_EQ(fl.get(160, 16), a);
}

template <typename FreeList>
void TestMerge() {
  std::mt19937_64 rng(12345);
  std::uniform_int_distribution<int> len_dist(1, 64);
  std::uniform_int_distribution<int> gap_dist(0, 32);
  std::bernoulli_distribution has_gap(0.25);
  std::uniform_int_distribution<int> which_list(0, 1);
  for (int iter = 0; iter < 10; iter++) {
    FreeList lists[2], ref;
    static char a alignas(16)[100000];
    int pos = 0;
    for (int i = 0; i < 1000; i++) {
      if (has_gap(rng))
        pos += gap_dist(rng);
      int l = len_dist(rng);
      lists[which_list(rng)].put(a + pos, l);
      ref.put(a + pos, l);
      pos += l;
    }
    lists[0].merge(std::move(lists[1]));
    lists[0].CheckEqual(ref);
  }
}


TEST(MMCoalescingFreeList, PutGet) {
  TestCoalescingPutGet<coalescing_free_list>();
}

TEST(MMCoalescingFreeTree, PutGet) {
  TestCoalescingPutGet<coalescing_free_tree>();
}

TEST(MMBestFitFreeList, RemoveIf) {
  best_fit_free_list fl;
  char storage alignas(16)[4000];
  char *a = storage + 128;   // thank you, overzealous compiler!
  fl.put(a, 100);
  fl.put(a + 100, 200);
  fl.put(a + 300, 700);
  ASSERT_FALSE(fl.remove_if_in_list(a - 100, 1));
  ASSERT_FALSE(fl.remove_if_in_list(a, 1));
  ASSERT_FALSE(fl.remove_if_in_list(a + 100, 1));
  ASSERT_FALSE(fl.remove_if_in_list(a + 300, 1));
  ASSERT_FALSE(fl.remove_if_in_list(a + 1000, 1));
  ASSERT_FALSE(fl.remove_if_in_list(a + 30, 70));

  ASSERT_TRUE(fl.remove_if_in_list(a + 100, 200));
  ASSERT_TRUE(fl.remove_if_in_list(a + 300, 700));
  ASSERT_TRUE(fl.remove_if_in_list(a, 100));
}

TEST(MMCoalescingFreeList, RemoveIf) {
  TestCoalescingRemoveIf<coalescing_free_list>();
}

TEST(MMCoalescingFreeTree, RemoveIf) {
  TestCoalescingRemoveIf<coalescing_free_tree>();
}

TEST(MMCoalescingFreeTree, Contains) {
  coalescing_free_tree fl;
  char a alignas(16)[4000];
  fl.put(a, 63);
  fl.put(a + 64, 2);
  EXPECT_TRUE(fl.contains(a, a + 63));
  EXPECT_TRUE(fl.contains(a + 10, a + 63));
  EXPECT_TRUE(fl.contains(a + 10, a + 13));
  EXPECT_TRUE(fl.contains(a + 64, a + 66));

  EXPECT_FALSE(fl.contains(a, a + 64));
  EXPECT_FALSE(fl.contains(a + 63, a + 65));
  EXPECT_FALSE(fl.contains(a + 64, a + 67));
}

TEST(MMBestFitFreeTree, Alignment) {
  best_fit_free_tree fl;
  fl.max_padding_ratio = 10;
  // workaround for large alignment bug in GCC
  char unaligned[4096+1024];
  char *storage = detail::align_ptr(unaligned, 1024);
  fl.put(storage+1, 1024);
  EXPECT_EQ(fl.get(1024, 512), nullptr) << "Block out of range or misaligned";
  fl.put(storage+1025, 2047);
  EXPECT_EQ(fl.get(1024, 512), storage + 1536);
}


TEST(MMCoalescingFreeTree, Alignment) {
  coalescing_free_tree fl;
  // workaround for large alignment bug in GCC
  char unaligned[4096+1024];
  char *storage = detail::align_ptr(unaligned, 1024);
  fl.put(storage+1, 1024);
  EXPECT_EQ(fl.get(1024, 512), nullptr) << "Block out of range or misaligned";
  fl.put(storage+1025, 2047);
  EXPECT_EQ(fl.get(1024, 512), storage + 512);
}

TEST(MMBestFitFreeTree, PaddingLimit) {
  best_fit_free_tree fl;
  fl.max_padding_ratio = 1.5;
  char storage alignas(16)[1024];
  fl.put(storage, 1024);
  EXPECT_EQ(fl.get(512,        16), nullptr) << "Excessive padding - should fail!";
  EXPECT_EQ(fl.get(1024*2/3,   16), nullptr) << "Excessive padding - should fail!";
  EXPECT_EQ(fl.get(1024*2/3+1, 16), storage) << "Should return the only block.";
}


TEST(MMBestFitFreeTree, BestFit) {
  best_fit_free_tree fl;
  fl.max_padding_ratio = 1.5;
  char storage alignas(16)[4096];
  fl.put(storage, 1024);
  fl.put(storage+1024, 256);
  fl.put(storage+1280, 2048);
  EXPECT_EQ(fl.get(1, 1), nullptr) << "Too much padding - should fail";
  char *p1 = static_cast<char*>(fl.get(200, 1));
  EXPECT_EQ(p1, storage+1024);
  EXPECT_EQ(fl.get(200, 1), nullptr) << "Good block already spent";
  char *p2 = static_cast<char *>(fl.get(1000, 1));
  EXPECT_EQ(p2, storage) << "Not a best-fitting block";
  EXPECT_EQ(fl.get(1000, 16), nullptr) << "Good block already spent";
  char *p3 = static_cast<char *>(fl.get(2000, 1));
  EXPECT_EQ(p3, storage + 1280) << "Not the only remaining block.";
}

TEST(MMBestFitFreeTree, RestorePadding) {
  best_fit_free_tree fl;
  fl.max_padding_ratio = 2;
  char storage alignas(16)[1024];
  fl.put(storage, 1024);
  void *ptr = fl.get(512, 512);
  EXPECT_TRUE(detail::is_aligned(ptr, 512)) << "Block not aligned";
  EXPECT_GE(static_cast<char*>(ptr), storage) << "Block out of bounds";
  EXPECT_LE(static_cast<char*>(ptr) + 512, storage + 1024) << "Block out of bounds";
  fl.put(ptr, 512);
  EXPECT_NE(fl.get(1024, 16), nullptr)
    << "Cannot allocate a block of original block size - padding not restored?";
}

TEST(MMBestFitFreeTree, RemoveIf) {
  best_fit_free_tree fl;
  char storage alignas(16)[4000];
  char *a = storage + 128;   // thank you, overzealous compiler!
  fl.put(a + 0, 1024);
  fl.put(a + 1024, 512);
  fl.put(a + 1536, 1024);
  ASSERT_FALSE(fl.remove_if_in_list(storage, 128));
  ASSERT_FALSE(fl.remove_if_in_list(a, 512));
  ASSERT_FALSE(fl.remove_if_in_list(a, 1536));
  ASSERT_FALSE(fl.remove_if_in_list(a + 1024, 1536));
  ASSERT_TRUE(fl.remove_if_in_list(a + 1024, 512));
  ASSERT_TRUE(fl.remove_if_in_list(a + 1536, 1024));
  ASSERT_TRUE(fl.remove_if_in_list(a + 0, 1024));
}


class test_coalescing_free_list : public coalescing_free_list {
 public:
  void CheckEqual(const test_coalescing_free_list &ref) {
    for (auto *blk = head_, *ref_blk = ref.head_;
        blk || ref_blk;
        blk = blk->next, ref_blk = ref_blk->next) {
      ASSERT_TRUE(blk) << "Output list too short";
      ASSERT_TRUE(ref_blk) << "Output list too long";
      ASSERT_EQ(blk->start, ref_blk->start) << "Blocks differ: start address mismath";
      ASSERT_EQ(blk->end, ref_blk->end) << "Blocks differ: end address mismath";
    }
  }
};

class test_coalescing_free_tree : public coalescing_free_tree {
 public:
  void CheckEqual(const test_coalescing_free_tree &ref) {
    for (auto it1 = by_addr_.cbegin(), it2 = ref.by_addr_.cbegin();
         it1 != by_addr_.cend() || it2 != ref.by_addr_.cend();
         ++it1, ++it2) {
      ASSERT_NE(it1, by_addr_.cend()) << "Too few free blocks.";
      ASSERT_NE(it2, ref.by_addr_.cend()) << "Too many free blocks.";
      ASSERT_EQ(it1->first, it2->first) << "Blocks differ: start address mismatch";
      ASSERT_EQ(it1->second, it2->second) << "Blocks differ: size mismatch";
    }
    EXPECT_EQ(by_size_, ref.by_size_) << "By-size map differs";
  }
};

TEST(MMCoalescingFreeList, Merge) {
  TestMerge<test_coalescing_free_list>();
}

TEST(MMCoalescingFreeTree, Merge) {
  TestMerge<test_coalescing_free_tree>();
}

}  // namespace test
}  // namespace mm
}  // namespace dali
