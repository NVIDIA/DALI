// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include <vector>
#include <array>
#include <cassert>
#include "dali/kernels/kernel_req.h"
#include "dali/kernels/scratch.h"

namespace dali {
namespace kernels {

static_assert(align_up(0, 2) == 0, "0 aligned up to 2 is 0");
static_assert(align_up(1, 2) == 2, "1 aligned up to 2 is 2");
static_assert(align_up(2, 2) == 2, "2 aligned up to 2 is 2");
static_assert(align_up(0, 8) == 0, "0 aligned up to 8 is 0");
static_assert(align_up(1, 8) == 8, "1 aligned up to 8 is 8");
static_assert(align_up(2, 8) == 8, "2 aligned up to 8 is 8");
static_assert(align_up(3, 8) == 8, "3 aligned up to 8 is 8");
static_assert(align_up(4, 8) == 8, "4 aligned up to 8 is 8");
static_assert(align_up(5, 8) == 8, "5 aligned up to 8 is 8");
static_assert(align_up(6, 8) == 8, "6 aligned up to 8 is 8");
static_assert(align_up(7, 8) == 8, "7 aligned up to 8 is 8");
static_assert(align_up(8, 8) == 8, "8 aligned up to 8 is 8");
static_assert(align_up(9, 8) == 16, "9 aligned up to 8 is 16");

template <typename T>
void test_add(ScratchpadEstimator &E, AllocType type, size_t count, size_t align = alignof(T)) {
  size_t prev = E.sizes[static_cast<int>(type)];
  EXPECT_EQ(align&(align-1), 0) << "Alignment must be a power of 2";
  size_t base = align_up(prev, align);
  E.add<T>(type, count, align);
  EXPECT_EQ(E.sizes[static_cast<int>(type)], base + count*sizeof(T));
}

TEST(Scratch, Estimator) {
  ScratchpadEstimator E;
  test_add<float>(E, AllocType::Host, 9);
  test_add<char>(E, AllocType::Host, 1);
  test_add<char>(E, AllocType::Host, 1);
  test_add<double>(E, AllocType::Host, 2);

  test_add<char>(E, AllocType::GPU, 1);
  test_add<float>(E, AllocType::GPU, 9);
  test_add<char>(E, AllocType::GPU, 1);
  test_add<double>(E, AllocType::GPU, 2);
  EXPECT_EQ(E.sizes[static_cast<int>(AllocType::Host)], 56);
  EXPECT_EQ(E.sizes[static_cast<int>(AllocType::GPU)], 64);
}

TEST(Scratch, BumpAllocator) {
  const size_t size = 1024;
  std::aligned_storage<size, 64>::type storage;
  BumpAllocator allocator(reinterpret_cast<char*>(&storage), size);
  size_t n0 = 10, n1 = 20, n2 = 33;

  EXPECT_EQ(allocator.total(), size);
  EXPECT_EQ(allocator.avail(), size);
  EXPECT_EQ(allocator.used(), 0);
  auto *p0 = allocator.alloc(n0);
  auto *p1 = allocator.alloc(n1);
  auto *p2 = allocator.alloc(n2);

  EXPECT_EQ(allocator.avail(), size-(n0+n1+n2));
  EXPECT_EQ(allocator.used(), n0+n1+n2);
  EXPECT_EQ(p1-p0, n0);
  EXPECT_EQ(p2-p1, n1);
  EXPECT_EQ(allocator.total(), size) << "Total size should remain constant";

  BumpAllocator allocator2 = std::move(allocator);
  EXPECT_EQ(allocator.total(), 0) << "After move, allocator should be empty";

  auto *p3 = allocator2.alloc(size-(n0+n1+n2));
  EXPECT_EQ(p3-p0, n0+n1+n2);
  EXPECT_EQ(allocator2.total(), size) << "Total size should remain constant";
  EXPECT_EQ(allocator2.used(), size);
  EXPECT_EQ(allocator2.avail(), 0);
}

template <typename T>
bool is_aligned(T *ptr, size_t alignment = alignof(T)) {
  return intptr_t(ptr) % alignment == 0;
}

TEST(Scratch, PreallocatedScratchpad) {
  PreallocatedScratchpad pad;

  const size_t size = 256;
  const size_t num_allocs = (size_t)AllocType::Count;
  ASSERT_EQ(pad.allocs.size(), num_allocs);

  const size_t alignment = 64;
  alignas(alignment) char storage[num_allocs][size];
  for (size_t i = 0; i < num_allocs; i++)
    pad.allocs[i] = BumpAllocator(storage[i], sizeof(storage[i]));

  for (size_t i = 0; i < num_allocs; i++) {
    AllocType type = AllocType(i);
    ASSERT_TRUE(is_aligned(pad.allocs[i].next(), alignment))
      << "Misaligned storage #" << i << "\n";

    int *p0 = pad.Allocate<int>(type, 2);
    EXPECT_EQ(p0, reinterpret_cast<int*>(&storage[i]))
     << "First item should be allocated at the beginning of the storage area";
    EXPECT_TRUE(is_aligned(p0));
    EXPECT_EQ(pad.Allocate<char>(type, 1), reinterpret_cast<char*>(p0) + 2*sizeof(*p0));

    int *p1 = pad.Allocate<int>(type, 3);
    EXPECT_TRUE(is_aligned(p1));
    EXPECT_EQ(pad.Allocate<char>(type, 1), reinterpret_cast<char*>(p1) + 3*sizeof(*p1));

    double *p2 = pad.Allocate<double>(type, 4);
    EXPECT_TRUE(is_aligned(p2));
    EXPECT_EQ(pad.Allocate<char>(type, 1), reinterpret_cast<char*>(p2) + 4*sizeof(*p2));

    double *p3 = pad.Allocate<double>(type, 1);
    EXPECT_TRUE(is_aligned(p2));
    EXPECT_EQ(pad.Allocate<char>(type, 1), reinterpret_cast<char*>(p3) + 1*sizeof(*p3));
  }
}

TEST(Scratch, ScratchpadAllocator) {
  ScratchpadAllocator sa;
  const size_t N = ScratchpadAllocator::NumAllocTypes;
  int sizes[N];
  for (size_t i = 0; i < N; i++) {
    AllocType type = AllocType(i);
    sizes[i] = 1024 + 256 * i;
    sa.Reserve(type, sizes[i]);
  }
  auto s = sa.GetScratchpad();
  for (size_t i = 0; i < N; i++) {
    float margin = sa.Policy(static_cast<AllocType>(i)).Margin;
    EXPECT_GE(s.allocs[i].total(), sizes[i]) << "Memory block smaller than requested";
    EXPECT_LE(s.allocs[i].total(), sizes[i] * (1 + margin) + 64) << "Too much padding";
    EXPECT_EQ(s.allocs[i].used(), 0) << "New scratchpad should be unused";
  }
}

}  // namespace kernels
}  // namespace dali
